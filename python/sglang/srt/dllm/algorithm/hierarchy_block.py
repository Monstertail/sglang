from typing import Any, List, Optional

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class HierarchyBlock(DllmAlgorithm):
    """Fast-dLLM v2 hierarchical block decoding with token inheritance.

    Each step denoises only the first sub-block that still has masks, with
    logits shifted one position as in the AR checkpoint. A block that starts
    with a mask takes its first token from the last-position prediction of the
    request's previous block, carried per request in ``dllm_algo_state``.
    """

    supported_architectures = ("Fast_dLLM_QwenForCausalLM",)
    inherits_next_token = True

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        if self.block_size % self.sub_block_size != 0:
            raise ValueError(
                f"sub_block_size ({self.sub_block_size}) must divide "
                f"block_size ({self.block_size})"
            )
        if self.fdfo:
            # FDFO clears dllm_algo_state when a block resolves, which would
            # drop the inherited token.
            raise ValueError("HierarchyBlock does not support --dllm-fdfo")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        # Count the inherited first token as generated output.
        start_list = self._block_start_list(forward_batch)
        self._inherit_first_token(forward_batch, algo_states)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        has_masks = any(start < self.block_size for start in start_list)
        if has_masks:
            for _ in range(self.max_steps(self.block_size)):
                done = self.step(forward_batch, out.logits_output.full_logits, None)
                if all(done):
                    break
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        # The last forward saw every block fully resolved, so its final row
        # predicts the first token of each request's next block.
        full_logits = out.logits_output.full_logits
        next_states = (
            full_logits.view(batch_size, self.block_size, -1)[:, -1]
            .argmax(dim=-1)
            .tolist()
        )

        next_token_ids_list = []
        if has_masks:
            next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size)
            next_token_ids_list = [
                next_token_ids[i, start_list[i] :] for i in range(batch_size)
            ]
        return (
            out.logits_output,
            next_token_ids_list,
            None,
            next_states,
            out.can_run_graph,
        )

    def _inherit_first_token(
        self, forward_batch: ForwardBatch, algo_states: Optional[List[Any]]
    ) -> None:
        if not algo_states or all(state is None for state in algo_states):
            return
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        inherited = torch.tensor(
            [-1 if state is None else state for state in algo_states],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        first = input_ids[:, 0]
        use = (first == self.mask_id) & (inherited >= 0)
        prompt_mask = forward_batch.dllm_prompt_mask
        if prompt_mask is not None:
            use &= ~prompt_mask[:, 0]
        # In place to keep the input_ids tensor identity (CUDA graph safe).
        first.copy_(torch.where(use, inherited, first))

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        logits = full_logits.view(batch_size, self.block_size, vocab_size)
        if self.token_shift > 0:
            # Position j is predicted by the logits at j - 1: [L0, L0, ..., LN-2].
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        masked = input_ids == self.mask_id
        prompt_mask = forward_batch.dllm_prompt_mask
        if prompt_mask is not None:
            masked &= ~prompt_mask
        done = ~masked.any(dim=1)

        # Only the first sub-block that still has masks is eligible.
        sub_has_mask = masked.view(batch_size, -1, self.sub_block_size).any(dim=2)
        active_sub = sub_has_mask.int().argmax(dim=1)
        sub_index = (
            torch.arange(self.block_size, device=input_ids.device)
            // self.sub_block_size
        )
        candidate = masked & (sub_index.unsqueeze(0) == active_sub.unsqueeze(1))

        preds = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(candidate, confidence, -float("inf"))

        # Accept tokens above the threshold, or fall back to the single most
        # confident candidate.
        transfer = confidence > self.threshold
        has_transfer = transfer.any(dim=1, keepdim=True)
        top1 = torch.zeros_like(transfer)
        top1.scatter_(1, confidence.argmax(dim=1, keepdim=True), True)
        transfer = torch.where(has_transfer, transfer, top1) & candidate

        new_input_ids = torch.where(transfer, preds, input_ids)
        forward_batch.input_ids.copy_(new_input_ids.view(-1))

        return done.tolist()


Algorithm = HierarchyBlock

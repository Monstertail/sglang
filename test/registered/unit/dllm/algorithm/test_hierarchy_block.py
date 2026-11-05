"""Unit tests for batched HierarchyBlock without a server or model weights."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm.hierarchy_block import HierarchyBlock
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import Req

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


MASK = 0
VOCAB_SIZE = 16
BLOCK_SIZE = 8
SUB_BLOCK_SIZE = 4


def make_config(threshold=0.5, fdfo=False):
    return DllmConfig(
        algorithm="HierarchyBlock",
        algorithm_config={"threshold": threshold, "sub_block_size": SUB_BLOCK_SIZE},
        block_size=BLOCK_SIZE,
        mask_id=MASK,
        max_running_requests=8,
        first_done_first_out_mode=fdfo,
        inherits_next_token=True,
    )


class FakeRunner:
    """Each row's logits depend only on that row's tokens, as with real attention."""

    def __init__(self, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(VOCAB_SIZE, VOCAB_SIZE, generator=generator) * 3
        self.mix = torch.randn(BLOCK_SIZE, BLOCK_SIZE, generator=generator)
        self.num_forwards = 0

    def logits_for(self, rows):
        return torch.einsum("pq,bqv->bpv", self.mix, self.embed[rows])

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.num_forwards += 1
        rows = forward_batch.input_ids.view(-1, BLOCK_SIZE)
        full_logits = self.logits_for(rows).reshape(-1, VOCAB_SIZE)
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=full_logits),
            can_run_graph=False,
        )


def make_batch(blocks):
    input_ids = torch.tensor(blocks, dtype=torch.long)
    return SimpleNamespace(
        batch_size=input_ids.shape[0],
        input_ids=input_ids.flatten(),
        dllm_prompt_mask=None,
    )


def reference_single(runner, block, threshold, inherited=None):
    """The single-request loop HierarchyBlock ran before batching."""
    ids = torch.tensor(block, dtype=torch.long)
    block_start = int((ids != MASK).sum())
    if ids[0] == MASK and inherited is not None:
        ids[0] = inherited
    for rel_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        rel_end = rel_start + SUB_BLOCK_SIZE
        while True:
            sub_mask = ids[rel_start:rel_end] == MASK
            if sub_mask.sum() == 0:
                break
            logits = runner.logits_for(ids.unsqueeze(0))[0]
            shifted = torch.cat([logits[:1], logits[:-1]], dim=0)[rel_start:rel_end]
            preds = shifted.argmax(dim=-1)
            conf = torch.softmax(shifted, dim=-1).gather(-1, preds.unsqueeze(-1))
            conf = torch.where(sub_mask, conf.squeeze(-1), float("-inf"))
            unmask = conf > threshold
            if unmask.sum() == 0:
                unmask[conf.argmax()] = True
            unmask &= sub_mask
            ids[rel_start:rel_end] = torch.where(unmask, preds, ids[rel_start:rel_end])
    next_token = runner.logits_for(ids.unsqueeze(0))[0, -1].argmax().item()
    return ids[block_start:].tolist(), next_token


class TestHierarchyBlockBatching(CustomTestCase):
    def test_batched_rows_match_single_request_loop(self):
        threshold = 0.5
        blocks = [
            [5, 6, 7, MASK, MASK, MASK, MASK, MASK],
            [MASK] * BLOCK_SIZE,
            [9, MASK, MASK, MASK, MASK, MASK, MASK, MASK],
        ]
        inherited = [None, 11, None]
        runner = FakeRunner()
        algo = HierarchyBlock(make_config(threshold=threshold))

        _, tokens, accept_lens, states, _ = algo.run(
            runner, make_batch(blocks), algo_states=inherited
        )

        self.assertIsNone(accept_lens)
        for row, block in enumerate(blocks):
            expected_tokens, expected_next = reference_single(
                runner, block, threshold, inherited=inherited[row]
            )
            self.assertEqual(tokens[row].tolist(), expected_tokens)
            self.assertEqual(states[row], expected_next)
        # The inherited token is reported as generated output.
        self.assertEqual(tokens[1][0].item(), 11)

    def test_prefill_only_round_emits_nothing_but_predicts_next_token(self):
        blocks = [[1, 2, 3, 4, 5, 6, 7, 8]]
        runner = FakeRunner()
        algo = HierarchyBlock(make_config())

        _, tokens, _, states, _ = algo.run(runner, make_batch(blocks))

        self.assertEqual(tokens, [])
        self.assertEqual(runner.num_forwards, 1)
        expected = runner.logits_for(torch.tensor(blocks))[0, -1].argmax().item()
        self.assertEqual(states, [expected])

    def test_only_first_masked_sub_block_is_denoised(self):
        algo = HierarchyBlock(make_config(threshold=0.0))
        batch = make_batch([[3, MASK, MASK, MASK, MASK, MASK, MASK, MASK]])
        logits = torch.zeros(BLOCK_SIZE, VOCAB_SIZE)
        logits[:, 4] = 10.0

        done = algo.step(batch, logits, None)

        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [3, 4, 4, 4] + [MASK] * 4)

    def test_rejects_fdfo(self):
        with self.assertRaises(ValueError):
            HierarchyBlock(make_config(fdfo=True))


class TestInheritedTokenPrefixBound(CustomTestCase):
    def _max_prefix_len(self, committed_len, inherits_next_token):
        req = SimpleNamespace(
            dllm_config=SimpleNamespace(
                block_size=BLOCK_SIZE,
                requires_separate_context_encoding=False,
                inherits_next_token=inherits_next_token,
            ),
            return_logprob=False,
            logprob_start_len=-1,
            origin_input_ids=list(range(committed_len)),
            output_ids=[],
            is_dllm=lambda: True,
        )
        return Req._compute_max_prefix_len(req, committed_len + BLOCK_SIZE)

    def test_last_committed_block_is_not_matched(self):
        # A block-aligned cache hit would skip the forward that predicts the
        # next block's first token.
        self.assertEqual(self._max_prefix_len(16, inherits_next_token=False), 16)
        self.assertEqual(self._max_prefix_len(16, inherits_next_token=True), 8)
        self.assertEqual(self._max_prefix_len(19, inherits_next_token=True), 16)


if __name__ == "__main__":
    unittest.main()

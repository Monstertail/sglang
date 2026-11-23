"""Forward hot reload as an SGLang general plugin, with no change to SGLang.

A running engine swaps its model forward for a candidate's, recaptures the
decode CUDA graphs and swaps back, without a restart. Install this package next
to SGLang and set ``SGLANG_FORWARD_HOT_RELOAD=1`` in the engine's environment;
without it the entry point does nothing. The ``sglang.srt.plugins`` entry point
then adds two scheduler RPCs, which these helpers call through
``Engine.collective_rpc``::

    from sglang_forward_hot_reload import hot_reload_forward, restore_forward

    engine = sgl.Engine(model_path=..., disable_prefill_cuda_graph=True)
    hot_reload_forward(engine, "candidate.py")  # {"active": <sha256>, "capture_s": ...}
    restore_forward(engine)                     # {"active": "original", ...}

``collective_rpc`` returns no value, so every TP rank writes its result as JSON
into a directory the helper creates, and the helper returns rank 0's.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

ENABLE_ENV = "SGLANG_FORWARD_HOT_RELOAD"
RPCS = ("hot_reload_forward", "restore_forward")


def register():
    """The ``sglang.srt.plugins`` entry point; SGLang runs it in every process."""
    if os.environ.get(ENABLE_ENV) != "1":
        return
    from sglang.srt.managers.scheduler import Scheduler

    from . import forward_hot_reload

    handlers = {name: getattr(forward_hot_reload, name) for name in RPCS}
    for name, handler in handlers.items():
        existing = getattr(Scheduler, name, None)
        if existing is not None and existing is not handler:
            raise RuntimeError(f"Scheduler already has {name!r}; not replacing it")
    for name, handler in handlers.items():
        setattr(Scheduler, name, handler)


def _call(engine, method: str, **kwargs) -> dict:
    with tempfile.TemporaryDirectory(prefix="sglang-forward-hot-reload-") as status:
        engine.collective_rpc(method, status_dir=status, **kwargs)
        return json.loads((Path(status) / "rank-0.json").read_text())


def hot_reload_forward(engine, path: str, sha256: str | None = None) -> dict:
    """Serve the forward that ``make_forward(original_forward)`` in ``path`` builds.

    Every scheduler loads the file, swaps the model's forward and recaptures its
    decode CUDA graphs; the KV cache is flushed. ``sha256`` pins the file content
    (computed here when omitted). A refusal raises ``AssertionError`` with the
    scheduler's message and leaves the engine serving its previous forward.
    """
    path = os.path.abspath(path)
    if sha256 is None:
        sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _call(engine, "hot_reload_forward", path=path, sha256=sha256)


def restore_forward(engine) -> dict:
    """Serve the model's original forward again (see ``hot_reload_forward``)."""
    return _call(engine, "restore_forward")

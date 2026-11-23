"""Process state that code loaded into a running engine can leave behind.

Code that runs inside the engine's processes, such as a hot-reloaded forward
(see ``forward_hot_reload``), can change state that outlives it: torch's
precision and determinism flags, the default dtype, the current stream,
``sys.path``. Putting the original forward back restores none of it. The guard
fingerprints that state and names what moved; it records environment variables
by digest and never reports a value.

A moved environment variable is reported but not fatal, because libraries set
some lazily on first use.
"""

import hashlib
import os
import sys


def _read(flags, name, getter):
    try:
        flags[name] = getter()
    except Exception as exc:  # a flag this torch build lacks reads the same both times
        flags[name] = f"unreadable: {type(exc).__name__}"


def fingerprint(torch, environ, path) -> dict:
    flags = {}
    cuda, cudnn = torch.backends.cuda, torch.backends.cudnn
    _read(flags, "default_dtype", lambda: str(torch.get_default_dtype()))
    _read(flags, "matmul_precision", torch.get_float32_matmul_precision)
    _read(flags, "tf32_matmul", lambda: bool(cuda.matmul.allow_tf32))
    _read(
        flags,
        "fp16_reduced_precision_reduction",
        lambda: bool(cuda.matmul.allow_fp16_reduced_precision_reduction),
    )
    _read(
        flags,
        "bf16_reduced_precision_reduction",
        lambda: bool(cuda.matmul.allow_bf16_reduced_precision_reduction),
    )
    _read(flags, "tf32_cudnn", lambda: bool(cudnn.allow_tf32))
    _read(flags, "cudnn_deterministic", lambda: bool(cudnn.deterministic))
    _read(flags, "cudnn_benchmark", lambda: bool(cudnn.benchmark))
    _read(
        flags,
        "deterministic_algorithms",
        lambda: bool(torch.are_deterministic_algorithms_enabled()),
    )
    _read(flags, "grad_enabled", lambda: bool(torch.is_grad_enabled()))
    _read(flags, "inference_mode", lambda: bool(torch.is_inference_mode_enabled()))
    _read(
        flags,
        "stream",
        lambda: (
            int(torch.cuda.current_stream().cuda_stream)
            if torch.cuda.is_available()
            else None
        ),
    )
    return {
        "flags": flags,
        "env": {
            name: hashlib.sha256(value.encode()).hexdigest()[:12]
            for name, value in environ.items()
        },
        "path": hashlib.sha256("\0".join(path).encode()).hexdigest()[:12],
    }


def drift(before: dict, after: dict) -> dict:
    """What differs, by name: torch flags with both values, environment variables
    added, removed or changed, and ``sys.path``."""
    moved = {"flags": [], "env": [], "path": []}
    for name in sorted(set(before["flags"]) | set(after["flags"])):
        was, now = before["flags"].get(name), after["flags"].get(name)
        if was != now:
            moved["flags"].append(f"{name} {was} -> {now}")
    for name in sorted(set(before["env"]) | set(after["env"])):
        was, now = before["env"].get(name), after["env"].get(name)
        if was != now:
            change = "added" if was is None else "removed" if now is None else "changed"
            moved["env"].append(f"{name} {change}")
    if before["path"] != after["path"]:
        moved["path"].append("sys.path changed")
    return moved


class ProcessStateGuard:
    """Takes its reference with ``reference()`` once, after the engine has served
    on its original state, so what libraries set on first use is part of it."""

    def __init__(self, torch, environ=None, path=None):
        self.torch = torch
        self.environ = os.environ if environ is None else environ
        self.path = sys.path if path is None else path
        self.ref = None

    def now(self) -> dict:
        return fingerprint(self.torch, self.environ, self.path)

    def reference(self) -> None:
        if self.ref is None:
            self.ref = self.now()

    def moved(self):
        return None if self.ref is None else drift(self.ref, self.now())

    def verify(self) -> str:
        """Raise if a torch flag, the dtype, the stream or ``sys.path`` moved; the
        same text on every rank that is in the same state."""
        moved = self.moved()
        fatal = [] if moved is None else moved["flags"] + moved["path"]
        if fatal:
            raise RuntimeError("process state drifted: " + "; ".join(fatal))
        return "ok"

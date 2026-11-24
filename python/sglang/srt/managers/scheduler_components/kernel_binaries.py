"""Prebuilt kernel binaries for ``--kernel-profiles``: the FlyDSL and Triton disk
caches built offline, shipped with a manifest, loaded at startup, and kept closed
to compiles while serving.

FlyDSL keeps each compiled kernel as ``<dir>/<function>_<key>/<cache key>.pkl`` and
Triton as files under ``<dir>/<hash>/``; both look there before compiling, and
``FLYDSL_RUNTIME_RUN_ONLY=1`` turns a FlyDSL miss into an error instead of a
compile. A binary set is a directory holding both caches, ``flydsl/`` and
``triton/``, built by running the engine offline over the shapes it serves
(``benchmark/kernel_profiles/build_binaries.py``), with ``manifest.json``: the
compilers and target the entries were built for, and each entry's size and sha256.
In the ``--kernel-profiles`` file::

    "binaries": {"dir": "/kp/binaries", "serving": "warn", "verify": true}

The engine points ``FLYDSL_RUNTIME_CACHE_DIR`` and ``TRITON_CACHE_DIR`` into ``dir``
before its schedulers start, so every kernel the two compile, the original ones and
every profile's, loads from it. Admission checks the manifest against this process
and lists what startup still had to compile. After admission ``serving`` decides
what a compile costs: ``warn`` lets it happen and reports it; ``strict`` sets
``FLYDSL_RUNTIME_RUN_ONLY=1`` so a FlyDSL kernel raises instead (Triton has no such
switch: its compiles are reported), and refuses to start on a manifest problem or a
kernel startup had to compile. ``verify: false`` checks sizes only. The directory
must be writable: both compilers lock and stage files in it.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
FORMAT = 1
RUN_ONLY = "FLYDSL_RUNTIME_RUN_ONLY"
# Cache name: (its directory under the binary set, the variable that points at it).
CACHES = {"flydsl": "FLYDSL_RUNTIME_CACHE_DIR", "triton": "TRITON_CACHE_DIR"}
SERVING = ("warn", "strict")


def toolchain() -> dict:
    """What a cache entry is only valid for: the compilers and the target."""
    import flydsl
    import triton
    from flydsl.compiler.jit_function import _flydsl_key

    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = get_rocm_arch()
    except Exception:  # no device: a CPU-only build host
        arch = os.environ.get("FLYDSL_ARCH") or None
    return {
        "flydsl": flydsl.__version__,
        "flydsl_key": hashlib.sha256(_flydsl_key().encode()).hexdigest(),
        "triton": triton.__version__,
        "arch": arch,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path):
    for path in sorted((root / "flydsl").glob("*/*.pkl")):
        yield path
    for path in sorted((root / "triton").glob("*/*")):
        # Triton stages a write in tmp.* and renames it.
        if path.is_file() and not path.name.startswith("tmp."):
            yield path


def entries(root, digest: bool = True) -> Dict[str, dict]:
    """Every cached kernel file under ``root``, by path relative to it."""
    root = Path(root)
    out = {}
    for path in _files(root):
        entry = {"bytes": path.stat().st_size}
        if digest:
            entry["sha256"] = _sha256(path)
        out[path.relative_to(root).as_posix()] = entry
    return out


def kernel(name: str) -> str:
    """The kernel a cache entry belongs to."""
    cache, directory, filename = name.split("/", 2)
    if cache == "flydsl":
        return "flydsl:" + directory.rsplit("_", 1)[0]
    return "triton:" + filename.removeprefix("__grp__").split(".", 1)[0]


def write_manifest(root, **extra) -> dict:
    root = Path(root)
    manifest = {
        "format": FORMAT,
        "built_at": time.time(),
        "toolchain": toolchain(),
        "entries": entries(root),
        **extra,
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def shipped_flydsl_cache() -> Optional[str]:
    """The FlyDSL cache aiter ships precompiled, which it points FlyDSL at on import
    unless the variable is set: the base a binary set is built on."""
    try:
        import aiter
    except ImportError:
        return None
    path = os.path.join(os.path.dirname(aiter.__file__), "jit", "flydsl_cache")
    return path if os.path.isdir(path) else None


def _default(variable: str) -> Optional[str]:
    if variable == "FLYDSL_RUNTIME_CACHE_DIR":
        return shipped_flydsl_cache()
    try:
        from sglang.srt.environ import third_party_cache_defaults
    except ImportError:
        return None
    return third_party_cache_defaults().get(variable)


def configure(profiles_path: Optional[str]) -> Optional[str]:
    """Point the compilers at the binaries a ``--kernel-profiles`` file names. Runs
    in the launching process, before the schedulers start, so they inherit it; a
    cache directory set by hand, other than SGLang's or aiter's default, is a
    conflict."""
    if not profiles_path:
        return None
    with open(profiles_path) as f:
        binaries = json.load(f).get("binaries")
    if not binaries:
        return None
    root = Path(binaries["dir"]).resolve()
    for cache, variable in CACHES.items():
        current = os.environ.get(variable)
        if current and Path(current).resolve() != root / cache and current != _default(variable):
            raise ValueError(f"{variable}={current} but the kernel profiles' binaries are {root / cache}")
        (root / cache).mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(root / cache)
    return str(root)


class Binaries:
    """The binary set this engine loads its kernels from."""

    def __init__(self, spec: dict):
        self.root = Path(spec["dir"]).resolve()
        self.serving = spec.get("serving", "warn")
        if self.serving not in SERVING:
            raise ValueError(f"binaries serving is one of {SERVING}, not {self.serving!r}")
        self.verify = bool(spec.get("verify", True))
        self.manifest: Optional[dict] = None
        self.problems: List[str] = []
        self.compiled_at_startup: List[str] = []
        self._serving_from: set = set()

    def check(self) -> List[str]:
        """Compare the manifest with this process and the directory; what startup
        compiled into it is ``compiled_at_startup``. Returns the problems."""
        problems = []
        for cache, variable in CACHES.items():
            current = os.environ.get(variable)
            if not current or Path(current).resolve() != self.root / cache:
                problems.append(f"{variable} is {current!r}, not {self.root / cache}")
        path = self.root / MANIFEST
        if not path.exists():
            problems.append(f"no {MANIFEST} in {self.root}")
            listed = {}
        else:
            self.manifest = json.loads(path.read_text())
            listed = self.manifest.get("entries", {})
            built = self.manifest.get("toolchain", {})
            here = toolchain()
            for key in sorted(set(built) | set(here)):
                if built.get(key) != here.get(key):
                    problems.append(f"built with {key} {built.get(key)}, this process has {here.get(key)}")
        present = entries(self.root, digest=self.verify)
        for name, entry in listed.items():
            got = present.get(name)
            if got is None:
                problems.append(f"{name} is missing")
            elif got["bytes"] != entry["bytes"] or (
                self.verify and got.get("sha256") != entry.get("sha256")
            ):
                problems.append(f"{name} is not the one built")
        self.compiled_at_startup = sorted({kernel(n) for n in present if n not in listed})
        self._serving_from = set(present)
        self.problems = problems
        return problems

    def guard(self) -> None:
        """From now on: report compiles (``warn``) or refuse FlyDSL's (``strict``)."""
        if self.serving == "strict":
            os.environ[RUN_ONLY] = "1"

    def compiled_while_serving(self) -> List[str]:
        return sorted(
            {
                kernel(path.relative_to(self.root).as_posix())
                for path in _files(self.root)
                if path.relative_to(self.root).as_posix() not in self._serving_from
            }
        )

    def describe(self) -> dict:
        listed = (self.manifest or {}).get("entries", {})
        return {
            "dir": str(self.root),
            "serving": self.serving,
            "run_only": os.environ.get(RUN_ONLY) == "1",
            "entries": len(listed),
            "bytes": sum(e["bytes"] for e in listed.values()),
            "problems": self.problems[:20],
            "compiled_at_startup": self.compiled_at_startup[:50],
            "compiled_while_serving": self.compiled_while_serving()[:50],
        }

"""Load weights synchronously, in every process of the server (on PYTHONPATH).

SGLang's DeepSeek-family loader copies host weights to the GPUs from a thread pool;
on MI355X with ROCm 7.2 those copies can deadlock inside HIP before the server is
ready. Serving is unaffected; only loading is serialized. KP_SYNC_WEIGHT_LOAD=0
turns it off.
"""

import importlib.abc
import importlib.util
import os
import sys

TARGET = "sglang.srt.model_loader.utils"


class _Synchronous(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        inner = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return inner.create_module(spec)

            def exec_module(self, module):
                inner.exec_module(module)
                module.should_async_load = lambda weight: False

        spec.loader = Loader()
        return spec


if os.environ.get("KP_SYNC_WEIGHT_LOAD", "1") == "1":
    sys.meta_path.insert(0, _Synchronous())

"""Scheduler RPC methods registered by plugins.

``Engine.collective_rpc(method, **kwargs)`` runs ``method`` on every scheduler
process of the engine. Built-in methods are ``Scheduler`` attributes; a plugin
registers its own here instead of patching ``Scheduler``::

    from sglang.srt.plugins.rpc_registry import register_rpc

    @register_rpc("my_plugin_reset")
    def my_plugin_reset(scheduler, *, level: int = 0):
        ...

The handler receives the scheduler and the call's keyword arguments. A string
it returns becomes the ``RpcReqOutput`` message, which ``collective_rpc``
returns to the caller; only the rank that answers the RPC is heard.
"""

from collections.abc import Callable
from typing import Optional

_RPC_HANDLERS: dict[str, Callable] = {}


def register_rpc(name: str, handler: Optional[Callable] = None):
    """Register ``handler`` as the scheduler RPC ``name``; usable as a decorator."""

    def _register(fn: Callable) -> Callable:
        if not callable(fn):
            raise TypeError(f"RPC handler for {name!r} must be callable")
        existing = _RPC_HANDLERS.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(f"RPC {name!r} is already registered by {existing!r}")
        _RPC_HANDLERS[name] = fn
        return fn

    if handler is not None:
        return _register(handler)
    return _register


def get_rpc(name: str) -> Optional[Callable]:
    return _RPC_HANDLERS.get(name)


def unregister_rpc(name: str) -> None:
    _RPC_HANDLERS.pop(name, None)

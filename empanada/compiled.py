"""torch.compile wrapper for helpers that used to be torch.jit.script."""

import functools

import torch

# Raised by dynamo/inductor when a function cannot be compiled. A normal
# error from the function itself (ValueError, shape mismatch) must still
# propagate, so only these compiler failures fall back to eager execution.
_COMPILE_FAILURE_NAMES = {
    "BackendCompilerFailed",
    "TorchDynamoException",
    "Unsupported",
    "InternalTorchDynamoError",
    "InductorError",
    "CppCompileError",
    "CompilerError",
}


def _is_compile_failure(exc):
    """Return True when ``exc`` came from torch.compile, not from ``fn``."""
    name = type(exc).__name__
    if name in _COMPILE_FAILURE_NAMES:
        return True
    module = type(exc).__module__ or ""
    if module.startswith("torch._dynamo") or module.startswith("torch._inductor"):
        return True
    return isinstance(exc, RuntimeError) and "compil" in str(exc).lower()


def compile_fn(fn):
    """Compile a tensor function with ``torch.compile``.

    TorchScript (``torch.jit.script``) is deprecated. ``torch.compile`` is
    the supported replacement for these helpers. ``torch.export`` is not:
    instance grouping branches on how many objects are in the image, and
    export rejects that data-dependent control flow.

    PyTorch older than 2.0, and builds whose inductor backend has no C++
    toolchain, run the original function instead.
    """
    compile_impl = getattr(torch, "compile", None)
    if compile_impl is None:
        return fn

    try:
        compiled = compile_impl(fn, dynamic=True)
    except TypeError:
        compiled = compile_impl(fn)

    state = {"compiled": True}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not state["compiled"]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if not _is_compile_failure(exc):
                raise
            state["compiled"] = False
            return fn(*args, **kwargs)

    return wrapper

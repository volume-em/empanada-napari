"""torch.compile wrapper for helpers that used to be torch.jit.script."""

import functools
import sys

import torch

# Raised by dynamo/inductor when a function cannot be compiled. 
# A normal error from the function itself (ValueError, shape mismatch) must still propagate
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


def _triton_would_clash():
    """Return True when importing triton now would crash the process.

    ``torch.compile`` imports torch._dynamo, which imports triton when it is
    installed. libtriton bundles its own LLVM. If another libLLVM is already
    mapped (Mesa's OpenGL driver loads one once napari opens its canvas, e.g.
    on WSLg or llvmpipe), loading libtriton segfaults. Loading triton before
    the GL driver is fine, so only check while triton is not yet loaded.
    """
    if "triton._C.libtriton" in sys.modules:
        return False
    try:
        with open("/proc/self/maps") as maps:
            return any("libLLVM" in line for line in maps)
    except OSError:
        # Not Linux: no /proc, and no Mesa LLVM driver to clash with.
        return False
        

def compile_fn(fn):
    """Compile a tensor function with ``torch.compile``.

    TorchScript (``torch.jit.script``) is deprecated. ``torch.compile`` is
    the supported replacement for these helpers. ``torch.export`` is not:
    instance grouping branches on how many objects are in the image, and
    export rejects that data-dependent control flow.

    Compilation is deferred to the first call, so importing this module
    never imports torch._dynamo (and triton). PyTorch older than
    2.0, builds whose inductor backend has no C++ toolchain, and processes
    where loading triton would crash run the original function instead.
    """
    compile_impl = getattr(torch, "compile", None)
    if compile_impl is None:
        return fn

    state = {"compiled": None}

    def _get_compiled():
        if state["compiled"] is None:
            if _triton_would_clash():
                state["compiled"] = False
                return None
            try:
                state["compiled"] = compile_impl(fn, dynamic=True)
            except TypeError:
                state["compiled"] = compile_impl(fn)
        return state["compiled"] or None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        compiled = _get_compiled()
        if compiled is None:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if not _is_compile_failure(exc):
                raise
            state["compiled"] = False
            return fn(*args, **kwargs)

    return wrapper

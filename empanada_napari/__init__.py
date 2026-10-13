try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import platform
import torch
import torch.multiprocessing as mp

# Fix macOS (Darwin) child processes (e.g. the 3D inference matcher process,
# or DataLoader/training workers) forking a process that already has Cocoa /
# CoreFoundation loaded by napari's Qt GUI. A bare fork() in that state is
# unsafe and can silently hang the child (leaving the plugin's progress bar
# stuck forever). The 'spawn' start method avoids this by using fork+exec.
#
# This must run as early as possible (before any widget code has a chance to
# implicitly create a multiprocessing object, e.g. via a joblib/dask backend),
# since Python's multiprocessing context can only be set once per process.
# `force=True` guarantees 'spawn' wins even if something already set a
# default context before this module was imported.
if platform.system() == "Darwin":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

if torch.backends.quantized.engine in (None or 'none'):
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'

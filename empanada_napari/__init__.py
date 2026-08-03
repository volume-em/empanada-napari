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

from ._slice_inference import slice_dock_widget
from ._volume_inference import volume_dock_widget
from ._finetune import finetuning_dock_widget, get_info_dock_widget
from ._train import training_dock_widget
from ._register_model import register_model_dock_widget
from ._pick_patches import pick_patches_widget, store_dataset_widget
from ._merge_split_widget import (
    morph_labels_widget, delete_labels_widget, split_labels_widget,
    merge_labels_widget, jump_to_label_widget,
    find_next_available_label_widget
)
from ._export_batch_segs import export_batch_segs_widget
from ._label_counter_widget import label_counter_widget

__all__ = [
    'slice_dock_widget',
    'volume_dock_widget',
    'finetune_dock_widget',
    'training_dock_widget',
    'register_model_dock_widget',
    'get_info_dock_widget',
    'pick_patches_widget',
    'store_dataset_widget',
    'merge_labels_widget',
    'split_labels_widget',
    'delete_labels_widget',
    'morph_labels_widget',
    'jump_to_label_widget',
    'find_next_available_label_widget',
    'export_batch_segs_widget',
    'label_counter_widget'
]


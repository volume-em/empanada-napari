try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

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

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._slice_inference import slice_dock_widget
from ._volume_inference import volume_dock_widget
from ._register_model import register_model_dock_widget
from ._merge_split_widget import delete_labels_widget, split_labels_widget, merge_labels_widget
#from ._merge_split_widget import jump_to_label_widget, split_by_dist

__all__ = [
    'slice_dock_widget',
    'volume_dock_widget',
    'register_model_dock_widget',
    'merge_labels_widget',
    'split_labels_widget',
    'delete_labels_widget',
    #'jump_to_label_widget',
    #'split_by_dist',
]

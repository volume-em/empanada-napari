try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._slice_inference import slice_dock_widget
from ._volume_inference import volume_dock_widget
from ._register_model import register_model_dock_widget
#from ._experimental import multigpu_inference_dock_widget

__all__ = [
    'slice_dock_widget',
    'volume_dock_widget',
    'register_model_dock_widget'
    #'multigpu_inference_dock_widget'
]

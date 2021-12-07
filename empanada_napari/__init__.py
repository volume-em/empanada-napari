try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._test_model import test_dock_widget
from ._stack_inference import stack_inference_dock_widget
from ._orthoplane_inference import orthoplane_inference_dock_widget

__all__ = [
    'test_dock_widget',
    'stack_inference_dock_widget',
    'orthoplane_inference_dock_widget'
]
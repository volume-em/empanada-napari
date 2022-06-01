import os
import yaml
import shutil
import urllib.request
from glob import glob

import napari
from napari import Viewer
from napari.layers import Image, Shapes
from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit
from empanada_napari.utils import add_new_model

def register_model_widget():
    @magicgui(
        call_button='Register model',
        layout='vertical',
        model_name=dict(widget_type='LineEdit', value='', label='Model name'),
        config_file=dict(widget_type='FileEdit', value='', label='Model config file', mode='r', tooltip='location of .yaml model config'),
        model_file=dict(widget_type='FileEdit', value='', label='Model file (optional)', mode='r', tooltip='location of .pth model, file path or url'),
        model_quant_file=dict(widget_type='FileEdit', value='', label='Quantized model file (optional)', mode='r', tooltip='location of quantized .pth model, file path or url'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        model_name,
        config_file,
        model_file,
        model_quant_file
    ):
        config_file = str(config_file)
        model_file = False if str(model_file) == '.' else str(model_file)
        model_quant_file = False if str(model_quant_file) == '.' else str(model_quant_file)
        add_new_model(model_name, config_file, model_file, model_quant_file)

        print('Finished!')

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def register_model_dock_widget():
    return register_model_widget, {'name': 'Register new model'}

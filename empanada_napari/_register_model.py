import napari
from napari import Viewer
from napari.layers import Image, Shapes
from napari_plugin_engine import napari_hook_implementation

from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

def register_model_widget():
    import os
    import yaml
    import shutil
    import urllib.request
    from glob import glob
    from napari.qt.threading import thread_worker
    from empanada.config_loaders import read_yaml
    from empanada_napari.utils import get_configs

    # get list of all available model configs
    model_configs = get_configs()

    def valid_url_or_file(fp):
        valid = False
        try:
            f = urllib.request.urlopen(fp)
            valid = True
        except:
            # make sure it's an accessible file
            valid = os.path.isfile(fp)

        # if it makes it here, we're good
        return valid

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
        print(config_file, model_file, model_quant_file)

        assert model_name, f'Model name cannot be empty!'
        assert config_file.endswith('.yaml'), f'Model config must be .yaml, got {config_file}'

        empanada_dir = os.path.join(os.path.expanduser('~'), '.empanada')
        config_dir = os.path.join(empanada_dir, 'configs')

        # makes both dirs if needed
        os.makedirs(config_dir, exist_ok=True)
        if model_name in list(model_configs.keys()):
            raise Exception(f'Model with name {model_name} already exists!')

        # load the config file
        config = read_yaml(config_file)

        # validate the given model files
        if model_file:
            assert valid_url_or_file(model_file), \
            f"{model_file} not a valid file or url!"

            # overwrite the model file
            config['model'] = model_file
        else:
            assert valid_url_or_file(config['model']), \
            f"{config['model']} not a valid file or url!"

        if model_quant_file:
            assert valid_url_or_file(model_quant_file), \
            "{model_quant_file} not a valid file or url!"

            # overwrite the model file
            config['model_quantized'] = model_quant_file
        elif config['model_quantized'] is not None:
            assert valid_url_or_file(config['model_quantized']), \
            f"{config['model_quantized']} not a valid file or url!"

        # save the config file to .empanada
        with open(os.path.join(config_dir, f'{model_name}.yaml'), mode='w') as f:
            yaml.dump(config, f)

    print('Finished!')

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def register_model_dock_widget():
    return register_model_widget, {'name': 'Register new model'}

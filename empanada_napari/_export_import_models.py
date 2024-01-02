import glob
import os
import shutil
from urllib.parse import urlparse
import napari
import yaml
from empanada.config_loaders import load_config
from magicgui import magicgui
import zipfile
import urllib.request
from empanada_napari.utils import get_configs, add_new_model
import requests

model_configs = get_configs()


def export_model_widget():
    gui_params = dict(
        model_name=dict(widget_type='ComboBox', label='Model name', choices=list(model_configs.keys()),
                        value=list(model_configs.keys())[0], tooltip='model to export'),
        export_path=dict(widget_type='FileEdit', label='Export path', mode='d', tooltip='location to export model'),
    )

    @magicgui(
        call_button='Export model',
        layout='vertical',
        **gui_params
    )
    def widget(viewer: napari.viewer.Viewer, model_name, export_path):
        config_yaml = model_configs[model_name]
        config = load_config(config_yaml)

        model_path = config['model']
        parsed = urlparse(model_path)
        if parsed.scheme and parsed.netloc:
            print(f'Downloading model from {model_path}')
            loc = model_path
            model_path = os.path.join(export_path, model_name + '.pth')

            with open(model_path, 'wb') as f:
                resp = requests.get(loc, verify=False)
                f.write(resp.content)
        else:
            shutil.copy(model_path, export_path)

        out_path = os.path.join(export_path, model_name + '.empanada')

        new_model_path = os.path.basename(model_path)

        config['model'] = new_model_path
        new_yaml = os.path.join(export_path, model_name + '.yaml')

        with open(new_yaml, 'w') as f:
            f.write(yaml.dump(config))

        with zipfile.ZipFile(str(out_path), 'w') as f:
            f.write(new_yaml, os.path.basename(new_yaml))
            f.write(os.path.join(export_path, new_model_path), new_model_path)

        os.remove(new_yaml)
        os.remove(os.path.join(export_path, new_model_path))

        print(f'Model exported to {out_path}')

    return widget


def import_model_widget():
    gui_params = dict(
        model_name=dict(widget_type='LineEdit', label='New model name (no spaces)', value='NewModelImported',
                        tooltip='new model name'),
        import_path=dict(widget_type='FileEdit', label='Model file', mode='r', tooltip='".empanada" file to import'),
    )

    @magicgui(
        call_button='Import model',
        layout='vertical',
        **gui_params
    )
    def widget(viewer: napari.viewer.Viewer, model_name, import_path):
        tmp_folder = os.path.join(os.path.dirname(import_path), 'tmp')

        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder, exist_ok=True)

        with zipfile.ZipFile(import_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_folder)

        new_yaml = os.path.join(tmp_folder, os.path.basename(import_path).replace('.empanada', '.yaml'))
        new_model = os.path.join(tmp_folder, os.path.basename(import_path).replace('.empanada', '.pth'))

        if not os.path.isfile(new_model):
            new_model = glob.glob(os.path.join(tmp_folder, '*.p*'))[0]

        target_models_folder = os.path.join(os.path.expanduser('~'), '.empanada/models')
        os.makedirs(target_models_folder, exist_ok=True)
        shutil.copy(new_model, os.path.join(target_models_folder, model_name + '.pth'))
        target_models_name = os.path.join(target_models_folder, model_name + '.pth')

        add_new_model(model_name, new_yaml, target_models_name)
        shutil.rmtree(tmp_folder)

    return widget



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
import warnings

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
        warnings.filterwarnings("ignore")
        config_yaml = model_configs[model_name]
        config = load_config(config_yaml)

        model_path = config['model']
        quantized_path = config.get('model_quantized')
        parsed = urlparse(model_path)
        if quantized_path:
            quantized_parsed = urlparse(quantized_path)
            if quantized_parsed.scheme and quantized_parsed.netloc:
                print(f'Downloading quantized model from {quantized_path}')
                loc = quantized_path
                quantized_path = os.path.join(export_path, model_name + '_quantized.pth')

                with open(quantized_path, 'wb') as f:
                    resp = requests.get(loc, verify=False)
                    f.write(resp.content)
            else:
                shutil.copy(quantized_path, export_path)

            new_quantized_path = os.path.basename(quantized_path)

            config['model_quantized'] = new_quantized_path

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
            if quantized_path:
                f.write(os.path.join(export_path, new_quantized_path), new_quantized_path)

        os.remove(new_yaml)
        os.remove(os.path.join(export_path, new_model_path))

        if quantized_path:
            os.remove(os.path.join(export_path, new_quantized_path))

        print(f'Model exported to {out_path}')
        warnings.filterwarnings("default")

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
        warnings.filterwarnings("ignore")
        tmp_folder = os.path.join(os.path.dirname(import_path), 'tmp')

        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder, exist_ok=True)

        with zipfile.ZipFile(import_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_folder)

        new_yaml = os.path.join(tmp_folder, os.path.basename(import_path).replace('.empanada', '.yaml'))
        new_model = os.path.join(tmp_folder, os.path.basename(import_path).replace('.empanada', '.pth'))
        target_models_folder = os.path.join(os.path.expanduser('~'), '.empanada/models')

        quantized_target_model_path = None
        new_yaml_c = yaml.load(open(new_yaml, 'r'), Loader=yaml.FullLoader)
        if new_yaml_c.get('model_quantized'):
            new_model_q = new_yaml_c.get('model_quantized')
            new_model_q_path = os.path.join(tmp_folder, new_model_q)
            shutil.copy(new_model_q_path, os.path.join(target_models_folder, model_name + '_quantized.pth'))
            quantized_target_model_path = os.path.join(target_models_folder, model_name + '_quantized.pth')

        if not os.path.isfile(new_model):
            new_model = glob.glob(os.path.join(tmp_folder, '*.p*'))[0]


        os.makedirs(target_models_folder, exist_ok=True)
        shutil.copy(new_model, os.path.join(target_models_folder, model_name + '.pth'))
        target_models_name = os.path.join(target_models_folder, model_name + '.pth')

        add_new_model(model_name, new_yaml, target_models_name, quantized_target_model_path if quantized_target_model_path else False)
        shutil.rmtree(tmp_folder)
        print(f'Model imported to {target_models_name}')
        warnings.filterwarnings("default")

    return widget



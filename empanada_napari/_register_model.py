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
    from glob import glob
    from zipfile import ZipFile
    from napari.qt.threading import thread_worker
    from empanada.config_loaders import read_yaml

    @thread_worker
    def unzip_model_files(zip_file, destination):
        with ZipFile(zip_file, 'r') as handle:
            handle.extractall(destination)

        root_dir = destination
        while True:
            subdirs = [
                sd for sd in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, sd))
            ]
            if len(subdirs) == 0:
                break
            elif len(subdirs) == 1:
                root_dir = os.path.join(root_dir, subdirs[0])
            else:
                raise Exception(f'Too many subdirs in zip file!')

        return root_dir


    @magicgui(
        call_button='Register model',
        layout='vertical',
        model_name=dict(widget_type='LineEdit', value='', label='Model name'),
        zip_file=dict(widget_type='FileEdit', value='', label='Model Zip File', mode='r', tooltip='location of .zip model export from empanada'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        model_name,
        zip_file
    ):
        zip_file = str(zip_file)
        if not model_name:
            raise Exception(f'Model name cannot be empty!')

        if not zip_file.endswith('.zip'):
            raise Exception(f'Model to register must be a .zip file, got {zip_file}')

        empanada_dir = os.path.join(os.path.expanduser('~'), '.empanada')
        config_dir = os.path.join(empanada_dir, 'configs')
        model_dir = os.path.join(empanada_dir, 'models')
        if not os.path.isdir(empanada_dir):
            os.mkdir(empanada_dir)
            os.mkdir(config_dir)
            os.mkdir(model_dir)
        elif os.path.isfile(os.path.join(config_dir, f'{model_name}.yaml')):
            raise Exception(f'Model name {model_name} is already registered, pick something else.')

        # create a temporary directory
        tmp_dir = os.path.join(empanada_dir, 'tmp')
        os.mkdir(tmp_dir)

        def _process_files(*args):
            root_dir = args[0]

            # load the config file
            config_files = glob(os.path.join(root_dir, '*.yaml'))
            if not len(config_files) == 1:
                raise Exception("There should only be 1 .yaml file in zip!")
            config = read_yaml(config_files[0])

            # get new names for the .pth files
            pth_files = glob(os.path.join(root_dir, '*.pth'))
            for fp in pth_files:
                fn = os.path.basename(fp)
                if fn.endswith('_base_cpu.pth'):
                    new_fn = f'{model_name}_base_cpu.pth'
                    config_key = 'base_model_cpu'
                elif fn.endswith('_base_gpu.pth'):
                    new_fn = f'{model_name}_base_gpu.pth'
                    config_key = 'base_model_gpu'
                elif fn.endswith('_render_cpu.pth'):
                    new_fn = f'{model_name}_render_cpu.pth'
                    config_key = 'render_model_cpu'
                elif fn.endswith('_render_gpu.pth'):
                    new_fn = f'{model_name}_render_gpu.pth'
                    config_key = 'render_model_gpu'
                else:
                    raise Exception(f'Unrecognized model file {fp}')

                new_fp = os.path.join(model_dir, new_fn)
                os.rename(fp, new_fp)
                config[config_key] = new_fp

            # save the config file to configs
            with open(os.path.join(config_dir, f'{model_name}.yaml'), mode='w') as f:
                yaml.dump(config, f)

            # delete the temporary directory
            shutil.rmtree(tmp_dir)

        # unzip the file into the configs directory
        worker = unzip_model_files(zip_file, tmp_dir)
        worker.returned.connect(_process_files)
        worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def register_model_dock_widget():
    return register_model_widget, {'name': 'Register new model'}

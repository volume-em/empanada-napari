import sys
import yaml
import os
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

import napari
from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

def export_model(config, state_fpath, save_path):
    import torch
    MODEL_DIR = os.path.join(os.path.expanduser('~'), '.empanada')
    torch.hub.set_dir(MODEL_DIR)

    # load the state_dict
    state_dict = torch.load(state_fpath)['state_dict']
    model = torch.hub.load_state_dict_from_url(config['model_definition'])

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # split the model into base and render
    # and script them
    base_model = deepcopy(model)
    render_model = model.semantic_pr

    base_model = torch.jit.script(base_model)
    render_model = torch.jit.script(render_model)

    # save the model
    model_name = config['model_name']
    base_fpath = os.path.join(save_path, f'{model_name}_base.pth')
    render_fpath = os.path.join(save_path, f'{model_name}_render.pth')

    torch.jit.save(base_model, base_fpath)
    torch.jit.save(render_model, render_fpath)

    # update the config
    config['base_model_cpu'] = base_fpath
    config['base_model_gpu'] = base_fpath
    config['render_model_cpu'] = render_fpath
    config['render_model_gpu'] = render_fpath

    # leave model definition alone for now
    # assumes no finetuning of finetuned models

    # save the config file in dir and to empanada path
    with open(os.path.join(save_path, f'{model_name}.yaml'), mode='w') as f:
        yaml.dump(config, f)

    empanada_dir = os.path.join(os.path.expanduser('~'), '.empanada')
    config_dir = os.path.join(empanada_dir, 'configs')
    with open(os.path.join(config_dir, f'{model_name}.yaml'), mode='w') as f:
        yaml.dump(config, f)

def finetuning_widget():
    from glob import glob
    from napari.qt.threading import thread_worker
    from empanada_napari.utils import abspath, get_configs
    from empanada.config_loaders import load_config

    from torch.cuda import device_count

    main_config = abspath(__file__, 'finetune_config.yaml')
    logo = abspath(__file__, 'resources/empanada_logo.png')

    # get list of all available model configs
    model_configs = get_configs()

    @thread_worker
    def run_finetuning(config):
        from empanada_napari import finetune
        finetune.main(config)

        outpath = os.path.join(config['TRAIN']['model_dir'], config['config_name'])
        print(f'Finished finetuning. Weights saved to {outpath}')

        return

    gui_params = dict(
        model_name=dict(widget_type='LineEdit', label='Model name, no spaces', value='FinetunedModel'),
        train_dir=dict(widget_type='FileEdit', label='Train directory', mode='d', tooltip='location were annotated data is saved'),
        model_dir=dict(widget_type='FileEdit', label='Model directory', mode='d', tooltip='directory in which to save the model weights'),
        finetune_model=dict(widget_type='ComboBox', label='Model to finetune', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='model to use for finetuning'),
        finetune_layer=dict(widget_type='ComboBox', label='Finetunable layers', choices=['none', 'stage4', 'stage3', 'stage2', 'stage1', 'all'], value='none', tooltip='layers to finetune in the encoder'),
        iterations=dict(widget_type='SpinBox', value=100, min=100, max=5000, step=100, label='Iterations', tooltip='number of iterations for finetuning')
    )

    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Finetune model',
        layout='vertical',
        **gui_params
    )
    def widget(
        viewer: napari.viewer.Viewer,
        label_head,
        model_name,
        train_dir,
        model_dir,
        finetune_model,
        finetune_layer,
        iterations
    ):
        train_dir = str(train_dir)
        model_dir = str(model_dir)

        assert os.path.isdir(train_dir)

        # get number of images in train_dir
        n_imgs = len(glob(os.path.join(train_dir, '**/images/*')))
        if not n_imgs:
            raise Exception(f"No images found in {os.path.join(train_dir, '**/images/*')}")
        elif n_imgs < 16:
            raise Exception(f'Need 16 images for finetuning, got {n_imgs}')
        else:
            epochs = int(iterations // (n_imgs / 16)) + 1

        epochs = 1
        print(f'Found {n_imgs} images for finetuning. Training for {epochs} epochs.')

        # load the model config
        train_config = load_config(main_config)
        model_config = load_config(model_configs[finetune_model])
        model_config['model_name'] = model_name

        train_config['config_name'] = model_name

        # set train_config params from model_config
        train_config['DATASET']['class_names'] = {
            k: v for k,v in zip(model_config['labels'], model_config['class_names'])
        }
        train_config['DATASET']['norms'] = model_config['norms']

        train_config['TRAIN']['train_dir'] = train_dir
        train_config['TRAIN']['model_dir'] = model_dir
        train_config['TRAIN']['whole_pretraining'] = model_config['model_definition']
        train_config['TRAIN']['finetune_layer'] = finetune_layer
        train_config['TRAIN']['schedule_params']['epochs'] = epochs

        data_cls = 'SingleClassInstanceDataset' if len(model_config['labels']) == 1 else 'PanopticDataset'
        train_config['TRAIN']['dataset_class'] = data_cls

        def _run_export_and_register():
            export_model(model_config, os.path.join(model_dir, f'{model_name}_checkpoint.pth.tar'), model_dir)
            print(f'Exported model to {model_dir}')

        worker = run_finetuning(train_config)
        worker.returned.connect(_run_export_and_register)
        worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def finetuning_dock_widget():
    return finetuning_widget, {'name': 'Finetune a model'}

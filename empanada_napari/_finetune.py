import sys
import yaml
import os
import platform
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

import napari
from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

def finetuning_widget():
    from glob import glob
    from napari.qt.threading import thread_worker
    from empanada_napari.utils import abspath, get_configs
    from empanada.config_loaders import load_config

    from torch.cuda import device_count

    main_config = abspath(__file__, 'training/finetune_config.yaml')
    logo = abspath(__file__, 'resources/empanada_logo.png')

    # get list of all available model configs
    model_configs = get_configs()

    @thread_worker
    def run_finetuning(config):
        from empanada_napari import finetune
        finetune.main(config)

        outpath = os.path.join(config['TRAIN']['model_dir'], config['model_name'] + '.pth')
        print(f'Finished finetuning. Model saved to {outpath}')

        return

    gui_params = dict(
        model_name=dict(widget_type='LineEdit', label='Model name, no spaces', value='FinetunedModel'),
        train_dir=dict(widget_type='FileEdit', label='Train directory', mode='d', tooltip='location were annotated training data is saved'),
        eval_dir=dict(widget_type='FileEdit', label='Validation directory', mode='d', tooltip='location were annotated validation data is saved'),
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
        eval_dir,
        model_dir,
        finetune_model,
        finetune_layer,
        iterations
    ):
        train_dir = str(train_dir)
        model_dir = str(model_dir)

        if not eval_dir:
            eval_dir = None

        assert os.path.isdir(train_dir)

        # get number of images in train_dir
        n_imgs = len(glob(os.path.join(train_dir, '**/images/*')))
        if not n_imgs:
            raise Exception(f"No images found in {os.path.join(train_dir, '**/images/*')}")
        elif n_imgs < 16:
            raise Exception(f'Need 16 images for finetuning, got {n_imgs}')
        else:
            epochs = int(iterations // (n_imgs / 16)) + 1

        print(f'Found {n_imgs} images for finetuning. Training for {epochs} epochs.')

        # load the model config
        config = load_config(main_config)
        model_config = load_config(model_configs[finetune_model])
        config['MODEL'] = {}
        for k,v in model_config.items():
            if k != 'FINETUNE':
                config['MODEL'][k] = model_config[k]
            else:
                config[k] = model_config[k]

        # training on mac breaks with more than 1 data worker
        if platform.system() == 'Darwin':
            config['TRAIN']['workers'] = 0

        config['model_name'] = model_name
        config['TRAIN']['train_dir'] = train_dir
        config['TRAIN']['model_dir'] = model_dir
        config['TRAIN']['finetune_layer'] = finetune_layer

        config['EVAL']['eval_dir'] = eval_dir

        if 'epochs' in config['TRAIN']['schedule_params']:
            config['TRAIN']['schedule_params']['epochs'] = epochs
        else:
            config['TRAIN']['epochs'] = epochs

        # fill labels to track for metrics
        for metric in config['TRAIN']['metrics']:
            if metric['metric'] in ['IoU', 'PQ']:
                metric['labels'] = config['MODEL']['labels']
            elif metric['metric'] in ['F1']:
                metric['labels'] = config['MODEL']['thing_list']
            else:
                raise Exception('Unsupported metric', metric['metric'])

        for metric in config['EVAL']['metrics']:
            if metric['metric'] in ['IoU', 'PQ']:
                metric['labels'] = config['MODEL']['labels']
            elif metric['metric'] in ['F1']:
                metric['labels'] = config['MODEL']['thing_list']
            else:
                raise Exception('Unsupported metric', metric['metric'])

        def _register_new_model():
            pass

        worker = run_finetuning(config)
        worker.returned.connect(_register_new_model)
        worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def finetuning_dock_widget():
    return finetuning_widget, {'name': 'Finetune a model'}

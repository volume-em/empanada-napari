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
    from empanada_napari.utils import abspath, get_configs, add_new_model
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

        outpath = os.path.join(config['TRAIN']['model_dir'], config['model_name'] + '.yaml')
        print(f'Finished finetuning. Model config saved to {outpath}')

        return outpath

    gui_params = dict(
        model_name=dict(widget_type='LineEdit', label='Model name, no spaces', value='FinetunedModel'),
        train_dir=dict(widget_type='FileEdit', label='Train directory', mode='d', tooltip='location were annotated training data is saved'),
        eval_dir=dict(widget_type='FileEdit', label='Validation directory (optional)', mode='d', tooltip='location were annotated validation data is saved'),
        model_dir=dict(widget_type='FileEdit', label='Model directory', mode='d', tooltip='directory in which to save the model weights'),
        finetune_model=dict(widget_type='ComboBox', label='Model to finetune', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='model to use for finetuning'),
        finetune_layer=dict(widget_type='ComboBox', label='Finetunable layers', choices=['none', 'stage4', 'stage3', 'stage2', 'stage1', 'all'], value='none', tooltip='layers to finetune in the encoder'),
        iterations=dict(widget_type='SpinBox', value=100, min=100, max=5000, step=100, label='Iterations', tooltip='number of iterations for finetuning'),
        patch_size=dict(widget_type='SpinBox', value=256, min=224, max=512, step=16, label='Patch size in pixels'),
        custom_config=dict(widget_type='FileEdit', label='Custom config (optional)', value='default config', tooltip='path to a custom empanada training config file; will not overwrite other parameters.'),
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
        iterations,
        patch_size,
        custom_config,
    ):
        train_dir = str(train_dir)
        model_dir = str(model_dir)

        if str(eval_dir) == '.':
            eval_dir = None

        assert os.path.isdir(train_dir)

        custom_config = str(custom_config)
        if custom_config != 'default config':
            assert os.path.isfile(custom_config)
            config = load_config(custom_config)
        else:
            config = load_config(main_config)

        # load the model config
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

        # get number of images in train_dir
        n_imgs = len(glob(os.path.join(train_dir, '**/images/*')))
        bsz = config['TRAIN']['batch_size']
        if not n_imgs:
            raise Exception(f"No images found in {os.path.join(train_dir, '**/images/*')}")
        elif n_imgs < bsz:
            raise Exception(f'Need {bsz} images for batch size {bsz}, got {n_imgs}.')
        else:
            epochs = int(iterations // (n_imgs // bsz))

        print(f'Found {n_imgs} images for training. Training for {epochs} epochs.')

        # update the patch size in augmentations if parameters are null
        for aug in config['TRAIN']['augmentations']:
            for k in aug.keys():
                aug[k] = patch_size if ('height' in k or 'width' in k) and aug.get(k) is None else aug[k]

        config['TRAIN']['save_freq'] = epochs // 5
        config['EVAL']['eval_dir'] = eval_dir
        # only run validation 5 times
        config['EVAL']['epochs_per_eval'] = epochs // 5

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

        def _register_new_model(outpath):
            add_new_model(model_name, outpath)

        worker = run_finetuning(config)
        worker.returned.connect(_register_new_model)
        worker.start()

    return widget

def get_info_widget():
    from empanada.config_loaders import load_config
    from empanada_napari.utils import get_configs
    model_configs = get_configs()

    gui_params = dict(
        model_name=dict(widget_type='ComboBox', label='Model name', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='model to get info for'),
    )
    @magicgui(
        call_button='Print info to terminal',
        layout='vertical',
        **gui_params
    )
    def widget(model_name):
        config = load_config(model_configs[model_name])

        # neatly print everything the user needs to read
        print('\n')
        print('MODEL INFORMATION')
        print('-----------------')
        print('Model name:', model_name)
        print('Description:\n', config.get('description'))

        # extract the class names list
        thing_list = config['thing_list']
        class_names = config['class_names']
        pf = config['padding_factor']

        ds_class = config['FINETUNE']['dataset_class']
        if ds_class == 'PanopticDataset':
            label_divisor = config['FINETUNE']['dataset_params']['label_divisor']
        else:
            label_divisor = None

        print('Finetuning instructions: \n')
        print(f'  The size of annotated patches should be divisible by {pf}')
        print(f'  Use a label divisor of {label_divisor}.')
        print(f'  Classes to annotate:')
        for cl,cn in class_names.items():
            kind = 'instance' if cl in thing_list else 'semantic'
            if label_divisor is not None:
                start_label = (cl * label_divisor) + 1
            else:
                start_label = 1

            print(f'    Class {cl} ({cn}) requires {kind} segmentation, start annotation at label {start_label}')

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def finetuning_dock_widget():
    return finetuning_widget, {'name': 'Finetune a model'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def get_info_dock_widget():
    return get_info_widget, {'name': 'Get model info'}

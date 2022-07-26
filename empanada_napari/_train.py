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

def training_widget():
    import yaml
    import torch
    from glob import glob
    from napari.qt.threading import thread_worker
    from empanada_napari.utils import abspath, get_configs, add_new_model
    from empanada.config_loaders import load_config
    from empanada.models import quantization as quant_models

    from torch.cuda import device_count

    main_config = abspath(__file__, 'training/train_config.yaml')
    cem_weights = "https://zenodo.org/record/6453160/files/cem1.5m_swav_resnet50_200ep_balanced.pth.tar?download=1"
    model_configs = {
        'PanopticDeepLab': abspath(__file__, 'training/pdl_model.yaml'),
        'PanopticBiFPN': abspath(__file__, 'training/bifpn_model.yaml')
    }
    logo = abspath(__file__, 'resources/empanada_logo.png')

    @thread_worker
    def run_training(config):
        from empanada_napari import train
        config = train.main(config)

        outpath = os.path.join(config['TRAIN']['model_dir'], config['model_name'] + '_checkpoint.pth.tar')
        print(f'Finished model training. Model weights saved to {outpath}')

        print(f'Scripting and exporting model...')
        model_arch = config['MODEL']['arch']
        quant_arch = 'Quantizable' + model_arch

        # load the state
        state = torch.load(outpath, map_location='cpu')
        norms = state['norms']
        state_dict = state['state_dict']

        # remove module. prefix from state_dict keys, if needed
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        model = quant_models.__dict__[quant_arch](**config['MODEL'], quantize=False)

        # prep the model
        model.load_state_dict(state_dict)
        model.eval()
        model.fuse_model()
        if torch.cuda.is_available():
            model.cuda()

        model = torch.jit.script(model)

        print('Model scripted successfully.')

        model_out = os.path.join(config['TRAIN']['model_dir'], f"{model_arch}_{config['model_name']}.pth")
        torch.jit.save(model, model_out)
        print('Exported model successfully.')

        # export a yaml file describing the models
        finetune_params = {
            'dataset_class': config['TRAIN']['dataset_class'],
            'dataset_params': config['TRAIN']['dataset_params'],
            'criterion': config['TRAIN']['criterion'],
            'criterion_params': config['TRAIN']['criterion_params'],
            'engine': config['EVAL']['engine'],
            'engine_params': config['EVAL']['engine_params'],
        }
        desc = {
            'model': model_out,
            'model_quantized': None,
            'norms': {'mean': norms['mean'], 'std': norms['std']},
            'padding_factor': 128,
            'thing_list': config['DATASET']['thing_list'],
            'labels': config['DATASET']['labels'],
            'class_names': config['DATASET']['class_names'],
            'description': config['description'],
            'FINETUNE': finetune_params
        }

        export_config_path = os.path.join(config['TRAIN']['model_dir'], f"{model_arch}_{config['model_name']}.yaml")
        with open(export_config_path, mode='w') as f:
            yaml.dump(desc, f)

        return export_config_path

    gui_params = dict(
        model_name=dict(widget_type='LineEdit', label='Model name, no spaces', value='NewModel'),
        train_dir=dict(widget_type='FileEdit', label='Train directory', mode='d', tooltip='location were annotated training data is saved'),
        eval_dir=dict(widget_type='FileEdit', label='Validation directory (optional)', mode='d', tooltip='location were annotated validation data is saved'),
        model_dir=dict(widget_type='FileEdit', label='Model directory', mode='d', tooltip='directory in which to save the model weights'),

        label_text=dict(widget_type='TextEdit', label='Dataset labels', tooltip='Separate line for each class. Each line must be {class_number},{class_name},{instance_or_semantic}'),
        label_divisor=dict(widget_type='LineEdit', label='Label divisor', value='1000', tooltip='Divisor used when annotating multiple classes. Ignored for single class instance segmentation.'),

        model_arch=dict(widget_type='ComboBox', label='Model architecture', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='Model architecture to train.'),
        use_cem=dict(widget_type='CheckBox', text='Use CEM pretrained weights', value=True, tooltip='Whether to initialize model with CEM pretrained weights.'),
        finetune_layer=dict(widget_type='ComboBox', label='Finetunable layers', choices=['none', 'stage4', 'stage3', 'stage2', 'stage1', 'all'], value='all', tooltip='Layers to finetune in the encoder. Ignored if not using CEM weights.'),
        iterations=dict(widget_type='SpinBox', value=500, min=1, max=10000, step=100, label='Iterations', tooltip='number of iterations for model training'),
        patch_size=dict(widget_type='SpinBox', value=256, min=224, max=512, step=16, label='Patch size in pixels'),
        custom_config=dict(widget_type='FileEdit', label='Custom config (optional)', value='default config', tooltip='path to a custom empanada training config file; will only overwrite the model architecture.'),

        description=dict(widget_type='TextEdit', label='Description (optional)', tooltip='Description of the model training data, purpose, links to more info, etc.'),
    )
    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Train model',
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

        label_text,
        label_divisor,

        model_arch,
        use_cem,
        finetune_layer,
        iterations,
        patch_size,
        custom_config,

        description
    ):
        train_dir = str(train_dir)
        model_dir = str(model_dir)

        if str(eval_dir) == '.':
            eval_dir = None

        assert os.path.isdir(train_dir)
        if model_arch == 'PanopticBiFPN':
            assert patch_size % 128 == 0, "Patch size must be divisible by 128 to use PanopticBiFPN!"

        # extract class_names, labels, and thing list
        class_names = {}
        thing_list = []
        for seg_class in label_text.split('\n'):
            class_id, class_name, class_kind = seg_class.split(',')
            class_id = class_id.strip()
            class_name = class_name.strip()
            class_kind = class_kind.strip()

            assert class_kind in ['semantic', 'instance'], "Class kind must be 'semantic' or 'instance'"
            class_names[int(class_id)] = class_name
            if class_kind == 'instance':
                thing_list.append(int(class_id))

        custom_config = str(custom_config)
        if custom_config != 'default config':
            assert os.path.isfile(custom_config)
            config = load_config(custom_config)
        else:
            config = load_config(main_config)

        # load the model config if needed
        if 'MODEL' not in config:
            config['MODEL'] = load_config(model_configs[model_arch])

        n_classes = len(class_names)
        config['MODEL']['num_classes'] = n_classes + 1 if n_classes > 1 else 1

        config['DATASET']['class_names'] = class_names
        config['DATASET']['labels'] = list(class_names.keys())
        config['DATASET']['thing_list'] = thing_list
        config['EVAL']['engine_params']['thing_list'] = thing_list

        config['description'] = description

        # training on mac breaks with more than 1 data worker
        if platform.system() == 'Darwin':
            config['TRAIN']['workers'] = 0

        config['model_name'] = model_name
        config['TRAIN']['train_dir'] = train_dir
        config['TRAIN']['model_dir'] = model_dir
        config['TRAIN']['finetune_layer'] = finetune_layer if use_cem else 'all'
        config['TRAIN']['encoder_pretraining'] = cem_weights if use_cem else None

        # turn off the instance decoder if all semantic channels
        if not thing_list:
            config['MODEL']['ins_decoder'] = False

        if n_classes == 1:
            config['TRAIN']['dataset_class'] = "SingleClassInstanceDataset"
        else:
            config['TRAIN']['dataset_class'] = "PanopticDataset"
            config['TRAIN']['dataset_params']['labels'] = config['DATASET']['labels']
            config['TRAIN']['dataset_params']['thing_list'] = config['DATASET']['thing_list']
            config['TRAIN']['dataset_params']['label_divisor'] = int(label_divisor)

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

        config['TRAIN']['save_freq'] = max(1, epochs // 5)
        config['EVAL']['eval_dir'] = eval_dir
        config['EVAL']['epochs_per_eval'] = max(1, epochs // 5)

        if 'epochs' in config['TRAIN']['schedule_params']:
            config['TRAIN']['schedule_params']['epochs'] = epochs
        else:
            config['TRAIN']['epochs'] = epochs

        # fill labels to track for metrics
        for metric in config['TRAIN']['metrics']:
            if metric['metric'] in ['IoU', 'PQ']:
                metric['labels'] = config['DATASET']['labels']
            elif metric['metric'] in ['F1']:
                metric['labels'] = config['DATASET']['thing_list']
            else:
                raise Exception('Unsupported metric', metric['metric'])

        for metric in config['EVAL']['metrics']:
            if metric['metric'] in ['IoU', 'PQ']:
                metric['labels'] = config['DATASET']['labels']
            elif metric['metric'] in ['F1']:
                metric['labels'] = config['DATASET']['thing_list']
            else:
                raise Exception('Unsupported metric', metric['metric'])

        def _register_new_model(outpath):
            add_new_model(model_name, outpath)
            print(f'Registered new model {model_name}')

        worker = run_training(config)
        worker.returned.connect(_register_new_model)
        worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def training_dock_widget():
    return training_widget, {'name': 'Train a model'}

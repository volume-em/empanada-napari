import os
import time
import yaml
import platform
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from empanada import losses
from empanada import data
from empanada import metrics
from empanada.inference import engines
from empanada.data.utils import FactorPad

from empanada_napari.utils import load_model_to_device

MODEL_DIR = os.path.join(os.path.expanduser('~'), '.empanada')
torch.hub.set_dir(MODEL_DIR)

schedules = sorted(name for name in lr_scheduler.__dict__
    if callable(lr_scheduler.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

optimizers = sorted(name for name in optim.__dict__
    if callable(optim.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

augmentations = sorted(name for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

datasets = sorted(name for name in data.__dict__
    if callable(data.__dict__[name])
)

engine_names = sorted(name for name in engines.__dict__
    if callable(engines.__dict__[name])
)

loss_names = sorted(name for name in losses.__dict__
    if callable(losses.__dict__[name])
)

def main(config):
    # create model directory if None
    if not os.path.isdir(config['TRAIN']['model_dir']):
        os.mkdir(config['TRAIN']['model_dir'])

    # validate parameters
    assert config['TRAIN']['lr_schedule'] in schedules
    assert config['TRAIN']['optimizer'] in optimizers
    assert config['FINETUNE']['criterion'] in loss_names
    assert config['FINETUNE']['engine'] in engine_names

    main_worker(config)

def main_worker(config):
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if str(config['device']) == 'cpu':
        print(f"Using CPU for training.")
    else:
        print(f"Using GPU for training.")

    if platform.system() == "Darwin":
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    # setup the model and pick dataset class
    model = load_model_to_device(config['MODEL']['model'], config['device'])
    norms = config['MODEL']['norms']
    dataset_class_name = config['FINETUNE']['dataset_class']
    data_cls = data.__dict__[dataset_class_name]

    finetune_layer = config['TRAIN']['finetune_layer']
    # start by freezing all encoder parameters
    for pname, param in model.named_parameters():
        if 'encoder' in pname:
            param.requires_grad = False

    # freeze encoder layers
    if finetune_layer == 'none':
        # leave all encoder layers frozen
        pass
    elif finetune_layer == 'all':
        # unfreeze all encoder parameters
        for pname, param in model.named_parameters():
            if 'encoder' in pname:
                param.requires_grad = True
    else:
        valid_layers = ['stage1', 'stage2', 'stage3', 'stage4']
        assert finetune_layer in valid_layers
        # unfreeze all layers from finetune_layer onward
        for layer_name in valid_layers[valid_layers.index(finetune_layer):]:
            # freeze all encoder parameters
            for pname, param in model.named_parameters():
                if f'encoder.{layer_name}' in pname:
                    param.requires_grad = True

    num_trainable = sum(p[1].numel() for p in model.named_parameters() if p[1].requires_grad)
    print(f'Model with {num_trainable} trainable parameters.')

    model = model.to(config['device'])

    cudnn.benchmark = True

    # set the training image augmentations
    config['aug_string'] = []
    dataset_augs = []
    for aug_params in config['TRAIN']['augmentations']:
        aug_name = aug_params['aug']

        assert aug_name in augmentations or aug_name == 'CopyPaste', \
        f'{aug_name} is not a valid albumentations augmentation!'

        config['aug_string'].append(aug_params['aug'])
        del aug_params['aug']
        dataset_augs.append(A.__dict__[aug_name](**aug_params))

    config['aug_string'] = ','.join(config['aug_string'])

    tfs = A.Compose([
        *dataset_augs,
        A.Normalize(**norms),
        ToTensorV2()
    ])

    # create training dataset and loader
    train_dataset = data_cls(config['TRAIN']['train_dir'], transforms=tfs, **config['FINETUNE']['dataset_params'])
    if config['TRAIN']['additional_train_dirs'] is not None:
        for train_dir in config['TRAIN']['additional_train_dirs']:
            add_dataset = data_cls(train_dir, transforms=tfs, **config['FINETUNE']['dataset_params'])
            train_dataset = train_dataset + add_dataset

    # num workers always less than number of batches in train dataset
    num_workers = min(config['TRAIN']['workers'], len(train_dataset) // config['TRAIN']['batch_size'])
    if platform.system() == "Darwin":
        num_workers = 0

    train_loader = DataLoader(
        train_dataset, batch_size=config['TRAIN']['batch_size'], shuffle=True,
        num_workers=config['TRAIN']['workers'], pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    if config['EVAL']['eval_dir'] is not None:
        eval_tfs = A.Compose([
            FactorPad(128), # pad image to be divisible by 128
            A.Normalize(**norms),
            ToTensorV2()
        ])
        eval_dataset = data_cls(config['EVAL']['eval_dir'], transforms=eval_tfs, **config['FINETUNE']['dataset_params'])
        # evaluation runs on a single gpu
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 pin_memory=torch.cuda.is_available(),
                                 num_workers=config['TRAIN']['workers'])
    else:
        eval_loader = None

    # set criterion
    criterion_name = config['FINETUNE']['criterion']
    criterion = losses.__dict__[criterion_name](**config['FINETUNE']['criterion_params']).to(config['device'])

    # set optimizer and lr scheduler
    opt_name = config['TRAIN']['optimizer']
    opt_params = config['TRAIN']['optimizer_params']
    optimizer = configure_optimizer(model, opt_name, **opt_params)

    schedule_name = config['TRAIN']['lr_schedule']
    schedule_params = config['TRAIN']['schedule_params']

    if 'steps_per_epoch' in schedule_params:
        n_steps = schedule_params['steps_per_epoch']
        if n_steps != len(train_loader):
            schedule_params['steps_per_epoch'] = len(train_loader)
            print(f'Steps per epoch adjusted from {n_steps} to {len(train_loader)}')

    scheduler = lr_scheduler.__dict__[schedule_name](optimizer, **schedule_params)
    scaler = GradScaler() if config['TRAIN']['amp'] else None

    config['start_epoch'] = 0
    if 'epochs' in config['TRAIN']['schedule_params']:
        epochs = config['TRAIN']['schedule_params']['epochs']
    elif 'epochs' in config['TRAIN']:
        epochs = config['TRAIN']['epochs']
    else:
        raise Exception('Number of training epochs not defined!')

    config['TRAIN']['epochs'] = epochs

    for epoch in range(config['start_epoch'], epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer,
              scheduler, scaler, epoch, config)

        # evaluate on validation set
        is_val_epoch = (epoch + 1) % config['EVAL']['epochs_per_eval'] == 0
        is_last_epoch = (epoch + 1) % epochs == 0
        if eval_loader is not None and (is_val_epoch or is_last_epoch):
            validate(eval_loader, model, criterion, epoch, config)

        save_now = (epoch + 1) % config['TRAIN']['save_freq'] == 0
        if save_now:
            outpath = os.path.join(config['TRAIN']['model_dir'], config['model_name'])
            torch.jit.save(model, outpath + '.pth')

            config['MODEL']['model'] = outpath + '.pth'
            config['MODEL']['model_quantized'] = None
            with open(outpath + '.yaml', mode='w') as f:
                yaml.dump({'FINETUNE': config['FINETUNE'], **config['MODEL']}, f)

def configure_optimizer(model, opt_name, **opt_params):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.

    Only BatchNorm parameters and biases are excluded.
    """

    # easy if there's no weight_decay
    if 'weight_decay' not in opt_params:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)
    elif opt_params['weight_decay'] == 0:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)

    decay = set()
    no_decay = set()
    param_dict = {}

    blacklist = (torch.nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = '%s.%s' % (mn, pn) if mn else pn

            if full_name.endswith('bias'):
                no_decay.add(full_name)
            elif full_name.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

            param_dict[full_name] = p

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert(len(inter_params) == 0), "Overlapping decay and no decay"
    assert(len(param_dict.keys() - union_params) == 0), "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    param_groups = [
        {"params": decay_params, **opt_params},
        {"params": no_decay_params, **opt_params}
    ]
    param_groups[1]['weight_decay'] = 0 # overwrite default to 0 for no_decay group

    return optim.__dict__[opt_name](param_groups, **opt_params)

def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    config
):
    # generic progress
    batch_time = ProgressAverageMeter('Time', ':6.3f')
    data_time = ProgressAverageMeter('Data', ':6.3f')
    loss_meters = None

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time],
        prefix="Epoch: [{}]".format(epoch)
    )

    # end of epoch metrics
    class_names = config['MODEL']['class_names']
    metric_dict = {}
    for metric_params in config['TRAIN']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        metric_params = {k: v for k,v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = metrics.__dict__[metric_name](metrics.EMAMeter, **metric_params)

    meters = metrics.ComposeMetrics(metric_dict, class_names)

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = batch['image']
        target = {k: v for k,v in batch.items() if k not in ['image', 'fname']}

        images = images.to(config['device'], non_blocking=True)
        target = {k: tensor.to(config['device'], non_blocking=True)
                  for k,tensor in target.items()}

        # zero grad before running
        optimizer.zero_grad()

        # compute output
        if scaler is not None:
            with autocast():
                output = model(images)
                loss, aux_loss = criterion(output, target)  # output and target are both dicts

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)
            loss, aux_loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # update the LR
        scheduler.step()

        # record losses
        if loss_meters is None:
            loss_meters = {}
            for k,v in aux_loss.items():
                loss_meters[k] = ProgressEMAMeter(k, ':.4e')
                loss_meters[k].update(v)
                # add to progress
                progress.meters.append(loss_meters[k])
        else:
            for k,v in aux_loss.items():
                loss_meters[k].update(v)

        # calculate human-readable per epoch metrics
        with torch.no_grad():
            meters.evaluate(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['TRAIN']['print_freq'] == 0:
            progress.display(i)

    # end of epoch print evaluation metrics
    print('\n')
    print(f'Epoch {epoch} training metrics:')
    meters.display()

def validate(
    eval_loader,
    model,
    criterion,
    epoch,
    config
):
    # validation metrics to track
    class_names = config['MODEL']['class_names']
    metric_dict = {}
    for metric_params in config['EVAL']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        metric_params = {k: v for k,v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = metrics.__dict__[metric_name](metrics.AverageMeter, **metric_params)

    meters = metrics.ComposeMetrics(metric_dict, class_names)

    # validation tracking
    batch_time = ProgressAverageMeter('Time', ':6.3f')
    loss_meters = None

    progress = ProgressMeter(
        len(eval_loader),
        [batch_time],
        prefix='Validation: '
    )

    # create the Inference Engine
    engine_name = config['FINETUNE']['engine']
    engine = engines.__dict__[engine_name](model, **config['FINETUNE']['engine_params'])

    for i, batch in enumerate(eval_loader):
        end = time.time()
        images = batch['image']
        target = {k: v for k,v in batch.items() if k not in ['image', 'fname']}

        images = images.to(config['device'], non_blocking=True)
        target = {k: tensor.to(config['device'], non_blocking=True)
                  for k,tensor in target.items()}

        # compute panoptic segmentations
        # from prediction and ground truth
        output = engine.infer(images)
        semantic = engine._harden_seg(output['sem'])
        output['pan_seg'] = engine.postprocess(
            semantic, output['ctr_hmp'], output['offsets']
        )
        target['pan_seg'] = engine.postprocess(
            target['sem'].unsqueeze(1), target['ctr_hmp'], target['offsets']
        )

        loss, aux_loss = criterion(output, target)

        # record losses
        if loss_meters is None:
            loss_meters = {}
            for k,v in aux_loss.items():
                loss_meters[k] = ProgressAverageMeter(k, ':.4e')
                loss_meters[k].update(v)
                # add to progress
                progress.meters.append(loss_meters[k])
        else:
            for k,v in aux_loss.items():
                loss_meters[k].update(v)

        # compute metrics
        with torch.no_grad():
            meters.evaluate(output, target)

        batch_time.update(time.time() - end)

        if i % config['TRAIN']['print_freq'] == 0:
            progress.display(i)

    # end of epoch print evaluation metrics
    print('\n')
    print(f'Validation results:')
    meters.display()
    print('\n')

class ProgressAverageMeter(metrics.AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        super().__init__()

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressEMAMeter(metrics.EMAMeter):
    """Computes and stores the exponential moving average and current value"""
    def __init__(self, name, fmt=':f', momentum=0.98):
        self.name = name
        self.fmt = fmt
        super().__init__(momentum)

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

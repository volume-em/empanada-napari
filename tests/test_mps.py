import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from empanada.inference.engines import _MedianQueue
from empanada.inference.postprocess import merge_semantic_and_instance
from empanada_napari import inference, finetune, train
from empanada_napari.utils import get_device


MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


@pytest.mark.parametrize('cuda,mps,use_gpu,expected', [
    (True, True, True, 'cuda:0'),
    (False, True, True, 'mps'),
    (False, False, True, 'cpu'),
    (True, True, False, 'cpu'),
    (False, True, False, 'cpu'),
    (False, None, True, 'cpu'),
], ids=['cuda_priority', 'mps', 'cpu_fallback', 'disable_cuda', 'disable_mps', 'older_torch'])
def test_get_device(cuda, mps, use_gpu, expected):
    backend = None if mps is None else SimpleNamespace(is_available=lambda: mps)
    with mock.patch('torch.cuda.is_available', return_value=cuda), \
         mock.patch.object(torch.backends, 'mps', backend, create=True):
        assert str(get_device(use_gpu)) == expected


@pytest.mark.parametrize('engine_class', [inference.Engine2d, inference.Engine3d],
                         ids=['2d', '3d'])
@pytest.mark.parametrize('use_gpu,quantized,expected_device,expected_model', [
    (True, True, 'mps', 'float.pth'),
    (False, True, 'cpu', 'quantized.pth'),
    (False, False, 'cpu', 'float.pth'),
], ids=['mps_uses_float', 'gpu_opt_out_quantized', 'gpu_opt_out_float'])
def test_engine_model_selection(engine_class, use_gpu, quantized, expected_device, expected_model):
    config = dict(model='float.pth', model_quantized='quantized.pth',
                  thing_list=[1], labels=[1], class_names={1: 'mito'},
                  padding_factor=16, norms=dict(mean=0.5, std=0.1))
    # Use the real device selector so ignoring use_gpu in either engine fails.
    with mock.patch('torch.cuda.is_available', return_value=False), \
         mock.patch.object(torch.backends, 'mps', SimpleNamespace(is_available=lambda: True), create=True), \
         mock.patch.object(inference, 'load_model_to_device') as load:
        engine_class(config, use_gpu=use_gpu, use_quantized=quantized)
        load.assert_called_once_with(expected_model, torch.device(expected_device))


@pytest.mark.parametrize('widget_name', ['slice', 'volume'])
@pytest.mark.parametrize('mps', [False, True], ids=['cpu', 'mps'])
def test_widget_device_defaults(widget_name, mps, qtbot):
    from empanada_napari import _slice_inference, _volume_inference

    module = _slice_inference if widget_name == 'slice' else _volume_inference
    factory = getattr(module, f'{widget_name}_inference_widget')
    with mock.patch('torch.cuda.is_available', return_value=False), \
         mock.patch.object(torch.backends, 'mps', SimpleNamespace(is_available=lambda: mps), create=True), \
         mock.patch.object(module, 'quantized_supported', True):
        widget = factory()
        qtbot.addWidget(widget.native)
        assert widget.use_gpu.value == mps
        assert widget.use_quantized.value == (not mps)


@pytest.mark.parametrize('module', [finetune, train], ids=['finetune', 'train'])
@pytest.mark.parametrize('cuda,mps,use_gpu,amp,expected_device,use_scaler', [
    (True, False, None, True, 'cuda:0', True),
    (True, False, None, False, 'cuda:0', False),
    (True, False, False, True, 'cpu', False),
    (False, True, None, True, 'mps', False),
], ids=['cuda_amp', 'amp_disabled', 'gpu_opt_out', 'mps_fp32'])
def test_training_device_and_amp(module, cuda, mps, use_gpu, amp, expected_device, use_scaler):
    norms = dict(mean=0.5, std=0.1)
    dataset_config = dict(dataset_class='SingleClassInstanceDataset',
                          dataset_params=dict(weight_gamma=None),
                          criterion='PanopticLoss', criterion_params={})
    config = dict(
        MODEL=dict(arch='PanopticBiFPNPR', model='unused.pth', norms=norms),
        DATASET=dict(norms=norms),
        FINETUNE=dataset_config,
        TRAIN=dict(
            **dataset_config, encoder_pretraining=None, finetune_layer='all',
            augmentations=[], train_dir='unused', additional_train_dirs=None,
            workers=0, batch_size=1, optimizer='SGD', optimizer_params=dict(lr=0.01),
            lr_schedule='StepLR', schedule_params=dict(step_size=1),
            amp=amp, epochs=1, save_freq=2),
        EVAL=dict(eval_dir=None, epochs_per_eval=1)
    )
    if use_gpu is not None:
        config['use_gpu'] = use_gpu
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    owner, factory = ((module, 'load_model_to_device') if module is finetune
                      else (module.models, 'PanopticBiFPNPR'))

    # Exercise worker setup, stopping at the epoch boundary. No GPU allocation,
    # dataset files, model downloads, or checkpoint writes are needed.
    with mock.patch('torch.cuda.is_available', return_value=cuda), \
         mock.patch.object(torch.backends, 'mps', SimpleNamespace(is_available=lambda: mps), create=True), \
         mock.patch.object(owner, factory, return_value=model), \
         mock.patch.object(model, 'to', return_value=model) as move_model, \
         mock.patch.object(module.data, 'SingleClassInstanceDataset', return_value=[{}]), \
         mock.patch.object(module, 'configure_optimizer', return_value=optimizer), \
         mock.patch.object(module, 'GradScaler') as scaler, \
         mock.patch.object(module, 'train') as epoch:
        module.main_worker(config)
        assert config['device'] == torch.device(expected_device)
        move_model.assert_called_once_with(torch.device(expected_device))
        epoch.assert_called_once()
        if use_scaler:
            scaler.assert_called_once_with()
            assert epoch.call_args.args[5] is scaler.return_value
        else:
            scaler.assert_not_called()
            assert epoch.call_args.args[5] is None


def test_cpu_majority_vote_ties():
    sem = torch.tensor([[[1, 2, 1, 2, 0, 0]]])
    ins = torch.tensor([[[1, 1, 2, 2, 0, 0]]])
    actual = merge_semantic_and_instance(sem, ins, 1000, [1, 2], 1, 0)
    expected = torch.tensor([[[1001, 1001, 1002, 1002, 0, 0]]])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('kernel', [1, 3, 5], ids=['identity', 'three_slices', 'five_slices'])
def test_cpu_median(kernel):
    values = torch.arange(kernel * 4, dtype=torch.float32).reshape(kernel, 1, 2, 2)
    queue = _MedianQueue(kernel)
    for value in values:
        queue.enqueue({'sem': value.unsqueeze(0)})
    assert torch.equal(queue.get_median('sem'), values[kernel // 2:kernel // 2 + 1])


@pytest.mark.skipif(not MPS_AVAILABLE, reason='Apple MPS unavailable')
def test_mps_median_large_image():
    # This shape reproduced incorrect MPS medians; small tensors missed the bug.
    values = torch.rand(3, 1, 2048, 2048, generator=torch.Generator().manual_seed(31))
    queue = _MedianQueue(3)
    for value in values:
        queue.enqueue({'sem': value.unsqueeze(0).to('mps')})
    actual = queue.get_median('sem')
    expected = torch.median(values, dim=0, keepdim=True).values
    assert actual.device.type == 'mps'
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not MPS_AVAILABLE, reason='Apple MPS unavailable')
def test_postprocessing_without_implicit_fallback():
    # The fallback flag is read at torch import, so use a fresh process.
    code = '''
import torch
from empanada.inference.postprocess import merge_semantic_and_instance
sem = torch.tensor([[[1, 2, 1, 2, 0, 0]]], device='mps')
ins = torch.tensor([[[1, 1, 2, 2, 0, 0]]], device='mps')
actual = merge_semantic_and_instance(sem, ins, 1000, [1, 2], 1, 0)
expected = torch.tensor([[[1001, 1001, 1002, 1002, 0, 0]]])
assert actual.device.type == 'mps'
assert torch.equal(actual.cpu(), expected)
'''
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0')
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=90)

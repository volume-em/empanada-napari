from unittest import mock

import torch
import torch.nn as nn

from empanada.inference.contiguous_encoder import (
    ContiguousEncoderFeatures,
    contiguous_features,
    wrap_contiguous_encoder_features,
)
from empanada.models.quantization.panoptic_bifpn import QuantizablePanopticBiFPN


class _BiFPNStub(nn.Module):
    r"""Minimal BiFPN-shaped module whose encoder returns strided tensors."""

    def _forward_encoder(self, x):
        # Channel-last then permute back: values preserved, storage strided.
        features = []
        for scale in (1, 2, 4, 8):
            fmap = x[:, :, : max(1, x.size(2) // scale), : max(1, x.size(3) // scale)]
            features.append(fmap.transpose(-1, -2))
        return features

    def p2_resample(self, p2):
        self.p2_was_contiguous = p2.is_contiguous()
        return p2

    def _forward_decoders(self, pyramid, p2_features):
        self.pyramid_contiguous = [t.is_contiguous() for t in pyramid]
        self.p2_decoder_contiguous = p2_features.is_contiguous()
        return p2_features, p2_features

    def _apply_heads(self, semantic_x, instance_x, render_steps=2, interpolate_ins=True):
        return {
            'sem_logits': semantic_x,
            'ctr_hmp': instance_x,
            'offsets': instance_x,
            'render_steps': render_steps,
            'interpolate_ins': interpolate_ins,
        }


class _DeepLabStub(nn.Module):
    def forward(self, x):
        return {'sem_logits': x}


def test_contiguous_features_preserves_values():
    strided = torch.randn(1, 4, 8, 8).permute(0, 1, 3, 2)
    assert not strided.is_contiguous()
    fixed = contiguous_features([strided])[0]
    assert fixed.is_contiguous()
    assert torch.equal(fixed, strided)


def test_wrapper_makes_encoder_maps_contiguous_before_decoder():
    model = _BiFPNStub()
    wrapped = wrap_contiguous_encoder_features(model)
    assert isinstance(wrapped, ContiguousEncoderFeatures)

    image = torch.randn(1, 3, 16, 16)
    out = wrapped(image, 3, interpolate_ins=False)

    encoder_out = model._forward_encoder(image)
    assert not encoder_out[1].is_contiguous()
    assert model.p2_was_contiguous
    assert model.p2_decoder_contiguous
    assert all(model.pyramid_contiguous)
    assert out['render_steps'] == 3
    assert out['interpolate_ins'] is False


def test_non_bifpn_models_are_not_wrapped():
    model = _DeepLabStub()
    assert wrap_contiguous_encoder_features(model) is model


def test_load_model_to_device_wraps_bifpn_script_models(tmp_path):
    from empanada_napari.utils import load_model_to_device

    weights = tmp_path / 'mini_quantized.pth'
    weights.write_bytes(b'placeholder')
    stub = _BiFPNStub()
    with mock.patch('empanada_napari.utils._load_torchscript', return_value=stub):
        loaded = load_model_to_device(str(weights), torch.device('cpu'))
    assert isinstance(loaded, ContiguousEncoderFeatures)
    assert loaded.model is stub


def test_python_quantized_encoder_returns_contiguous_maps():
    class _StridedEncoder(nn.Module):
        def forward(self, x):
            return [x.permute(0, 1, 3, 2)]

    model = QuantizablePanopticBiFPN.__new__(QuantizablePanopticBiFPN)
    nn.Module.__init__(model)
    model.quant = nn.Identity()
    model.dequant = nn.Identity()
    model.encoder = _StridedEncoder()

    image = torch.randn(1, 2, 4, 8)
    strided = model.encoder(image)[0]
    assert not strided.is_contiguous()
    features = QuantizablePanopticBiFPN._forward_encoder(model, image)
    assert features[0].is_contiguous()
    assert torch.equal(features[0], strided)


def test_slice_widgets_reuse_warmed_engine():
    from empanada_napari import _slice_inference as slice_inference

    slice_inference._ENGINE_CACHE.clear()
    created = []

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            created.append(kwargs.get('use_quantized'))

        def update_params(self, **kwargs):
            return None

    def make(use_quantized):
        widget = slice_inference.SliceInferenceWidget.__new__(
            slice_inference.SliceInferenceWidget
        )
        widget.model_config_name = 'MitoNet_v1_mini'
        widget.model_config = {}
        widget.using_gpu = False
        widget.using_quantized = use_quantized
        widget.downsampling = 1
        widget.min_distance_object_centers = 3
        widget.center_confidence_thr = 0.1
        widget.confidence_thr = 0.5
        widget.maximum_objects_per_class = 1000
        widget.semantic_only = False
        widget.fine_boundaries = False
        widget.tile_size = 0
        widget.engine = None
        widget.last_config = None
        widget.get_engine()
        return widget

    with mock.patch.object(slice_inference, 'Engine2d', _FakeEngine):
        first = make(True)
        second = make(True)
        fp32 = make(False)

    assert first.engine is second.engine
    assert fp32.engine is not first.engine
    assert created == [True, False]
    slice_inference._ENGINE_CACHE.clear()

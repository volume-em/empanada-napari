import torch.nn as nn

_ENCODER_PATH_ATTRS = (
    '_forward_encoder',
    'p2_resample',
    '_forward_decoders',
    '_apply_heads',
)


def contiguous_features(features):
    r"""Copy encoder tensors into a dense layout if they are strided.

    Quantized Mini models emit non-contiguous encoder maps; the FP32 decoder
    is much slower on that layout (see issue #81). Already-contiguous tensors
    are returned unchanged.
    """
    return [feature.contiguous() for feature in features]


def has_bifpn_encoder_path(model):
    return all(hasattr(model, name) for name in _ENCODER_PATH_ATTRS)


class ContiguousEncoderFeatures(nn.Module):
    r"""Replay a BiFPN forward, making encoder outputs contiguous first.

    Shipped models are TorchScript, so changing Python ``_forward_encoder``
    does not affect already-exported Mini checkpoints. This wrapper calls the
    scripted encoder/decoder pieces around a ``.contiguous()`` copy.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        features = contiguous_features(self.model._forward_encoder(x))
        p2_features = self.model.p2_resample(features[1])
        semantic_x, instance_x = self.model._forward_decoders(features[2:], p2_features)
        return self.model._apply_heads(semantic_x, instance_x, *args, **kwargs)

    def eval(self):
        self.model.eval()
        return super().eval()

    def train(self, mode=True):
        self.model.train(mode)
        return super().train(mode)


def wrap_contiguous_encoder_features(model):
    if isinstance(model, ContiguousEncoderFeatures):
        return model
    if not has_bifpn_encoder_path(model):
        return model
    return ContiguousEncoderFeatures(model)

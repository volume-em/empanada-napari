import pytest

torch = pytest.importorskip("torch")

from empanada.compiled import _is_compile_failure, compile_fn
from empanada.inference.postprocess import factor_pad


def test_factor_pad_matches_eager_padding():
    image = torch.zeros(1, 1, 10, 12)
    padded = factor_pad(image, 16)
    assert padded.shape == (1, 1, 16, 16)


def test_factor_pad_leaves_aligned_tensors_unchanged():
    image = torch.zeros(1, 1, 16, 32)
    padded = factor_pad(image, 16)
    assert padded.shape == image.shape
    assert padded.data_ptr() == image.data_ptr()


def test_compile_failure_falls_back_to_eager(monkeypatch):
    def add_one(tensor):
        return tensor + 1

    class BackendCompilerFailed(RuntimeError):
        pass

    def fail_then_unused(fn, dynamic=True):
        def compiled(*args, **kwargs):
            raise BackendCompilerFailed("inductor failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fail_then_unused)
    wrapped = compile_fn(add_one)
    result = wrapped(torch.zeros(2))
    assert torch.equal(result, torch.ones(2))
    assert _is_compile_failure(BackendCompilerFailed("inductor failed"))

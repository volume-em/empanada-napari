import io
import os
import urllib.error
import zipfile
from unittest import mock

import pytest

from empanada_napari.utils import _download_with_retries, _is_complete_zip, load_model_to_device


def _write_minimal_zip(path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("weights", b"ok")
    with open(path, "wb") as handle:
        handle.write(buffer.getvalue())


class TestDownloadWithRetries:
    r"""Regression tests for transient network errors (e.g. 502/504 gateway
    errors from a flaky CDN) when downloading model weights via
    ``load_model_to_device`` -> ``_download_with_retries``. Previously a
    single transient error would immediately fail inference/tests with no
    retry.
    """

    def test_recovers_after_transient_failures(self, tmp_path):
        dst = str(tmp_path / "model.pt")
        calls = {"n": 0}

        def fake_download(url, cached_file, hash_prefix, progress=True):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.HTTPError(url, 504, "Gateway Time-out", {}, None)
            _write_minimal_zip(cached_file)

        with mock.patch("torch.hub.download_url_to_file", side_effect=fake_download), \
             mock.patch("time.sleep", return_value=None) as mock_sleep:
            _download_with_retries(
                "http://example.com/model.pt", dst, max_retries=4, initial_backoff=0.01
            )

        assert calls["n"] == 3
        assert os.path.exists(dst)
        # slept between attempt 1->2 and 2->3, but not after the final success
        assert mock_sleep.call_count == 2

    def test_raises_after_exhausting_retries(self, tmp_path):
        dst = str(tmp_path / "model.pt")
        calls = {"n": 0}

        def always_fails(url, cached_file, hash_prefix, progress=True):
            calls["n"] += 1
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", {}, None)

        with mock.patch("torch.hub.download_url_to_file", side_effect=always_fails), \
             mock.patch("time.sleep", return_value=None):
            with pytest.raises(urllib.error.HTTPError):
                _download_with_retries(
                    "http://example.com/model.pt", dst, max_retries=3, initial_backoff=0.01
                )

        assert calls["n"] == 3
        assert not os.path.exists(dst)

    def test_cleans_up_partial_file_before_retrying(self, tmp_path):
        dst = str(tmp_path / "model.pt")
        calls = {"n": 0}

        def fake_download(url, cached_file, hash_prefix, progress=True):
            calls["n"] += 1
            if calls["n"] == 1:
                # simulate a partially-written file before the connection dropped
                with open(cached_file, "w") as f:
                    f.write("partial-garbage")
                raise urllib.error.URLError("connection reset")
            # on retry, the stale partial file must already be gone
            assert not os.path.exists(cached_file)
            _write_minimal_zip(cached_file)

        with mock.patch("torch.hub.download_url_to_file", side_effect=fake_download), \
             mock.patch("time.sleep", return_value=None):
            _download_with_retries(
                "http://example.com/model.pt", dst, max_retries=3, initial_backoff=0.01
            )

        assert calls["n"] == 2
        assert _is_complete_zip(dst)

    def test_non_retryable_error_propagates_immediately(self, tmp_path):
        dst = str(tmp_path / "model.pt")
        calls = {"n": 0}

        def raises_value_error(url, cached_file, hash_prefix, progress=True):
            calls["n"] += 1
            raise ValueError("not a network error")

        with mock.patch("torch.hub.download_url_to_file", side_effect=raises_value_error), \
             mock.patch("time.sleep", return_value=None):
            with pytest.raises(ValueError):
                _download_with_retries(
                    "http://example.com/model.pt", dst, max_retries=4, initial_backoff=0.01
                )

        assert calls["n"] == 1

    def test_retries_when_download_is_truncated(self, tmp_path):
        dst = str(tmp_path / "model.pt")
        calls = {"n": 0}

        def fake_download(url, cached_file, hash_prefix, progress=True):
            calls["n"] += 1
            if calls["n"] == 1:
                with open(cached_file, "wb") as handle:
                    handle.write(b"PK\x03\x04truncated")
                return
            _write_minimal_zip(cached_file)

        with mock.patch("torch.hub.download_url_to_file", side_effect=fake_download), \
             mock.patch("time.sleep", return_value=None):
            _download_with_retries(
                "http://example.com/model.pt", dst, max_retries=3, initial_backoff=0.01
            )

        assert calls["n"] == 2
        assert _is_complete_zip(dst)

    def test_load_replaces_truncated_cache(self, tmp_path):
        cached = tmp_path / "model.pth"
        cached.write_bytes(b"PK\x03\x04truncated")
        sentinel = object()

        def fake_download(url, cached_file, hash_prefix, progress=True):
            _write_minimal_zip(cached_file)

        with mock.patch("empanada_napari.utils.torch.hub.get_dir", return_value=str(tmp_path)), \
             mock.patch("torch.hub.download_url_to_file", side_effect=fake_download), \
             mock.patch("empanada_napari.utils.torch.jit.load", return_value=sentinel) as load:
            model = load_model_to_device("https://example.com/files/model.pth?download=1", "cpu")

        assert model is sentinel
        load.assert_called_once()
        assert _is_complete_zip(cached)

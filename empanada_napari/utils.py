import os, sys, time, yaml
import numpy as np
import requests
import torch
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import urlparse
from empanada.config_loaders import read_yaml
import warnings
import contextlib
import json
import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

__all__ = [
    'abspath'
    'get_configs',
    'Preprocessor',
    'enable_layer_rename_refresh'
]

MODEL_DIR = os.path.join(os.path.expanduser('~'), '.empanada')
torch.hub.set_dir(MODEL_DIR)

def abspath(root, relpath):
    root = Path(root)
    if root.is_dir():
        path = root/relpath
    else:
        path = root.parent/relpath
    return str(path.absolute())

def get_configs():
    # get dict of all model configs
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    model_configs = {}
    for fn in os.listdir(config_path):
        if fn.endswith('.yaml'):
            model_configs[fn[:-len('.yaml')]] = os.path.join(config_path, fn)

    empanada_path = os.path.join(os.path.expanduser('~'), '.empanada/configs')
    if os.path.isdir(empanada_path):
        for fn in os.listdir(empanada_path):
            if fn.endswith('.yaml'):
                model_configs[fn[:-len('.yaml')]] = os.path.join(empanada_path, fn)

    return model_configs

# Errors worth retrying: transient network/server hiccups (e.g. a 502/504
# from a CDN edge), as opposed to e.g. a 404 for a genuinely missing file.
_RETRYABLE_DOWNLOAD_ERRORS = (
    urllib.error.URLError,  # covers HTTPError (502/504/etc.) too
    ConnectionError,
    TimeoutError,
    OSError,
)


def _download_with_retries(url, cached_file, max_retries=4, initial_backoff=2.0):
    r"""Downloads a file via torch.hub, retrying transient network errors
    (e.g. 502/504 gateway errors from a flaky CDN) with exponential backoff
    before giving up.
    """
    hash_prefix = None
    for attempt in range(1, max_retries + 1):
        try:
            with no_ssl_verification():
                torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=True)
            return
        except _RETRYABLE_DOWNLOAD_ERRORS as exc:
            # remove any partially-downloaded file so a retry starts fresh
            if os.path.exists(cached_file):
                try:
                    os.remove(cached_file)
                except OSError:
                    pass

            if attempt == max_retries:
                raise

            wait_s = initial_backoff * (2 ** (attempt - 1))
            sys.stderr.write(
                f'Download of "{url}" failed ({exc}); retrying in '
                f'{wait_s:.0f}s (attempt {attempt}/{max_retries})...\n'
            )
            time.sleep(wait_s)


def load_model_to_device(fpath_or_url, device):
    # check whether local file or url
    if os.path.isfile(fpath_or_url):
        model = torch.jit.load(fpath_or_url, map_location=device)
    else:
        hub_dir = torch.hub.get_dir()

        # download file to hub_dir
        try:
            os.makedirs(hub_dir)
        except:
            pass

        # set the filename
        parts = urlparse(fpath_or_url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(hub_dir, filename)

        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(fpath_or_url, cached_file))
            _download_with_retries(fpath_or_url, cached_file)

        model = torch.jit.load(cached_file, map_location=device)

    return model

def valid_url_or_file(fp):
    valid = False
    try:
        f = requests.get(fp, verify=False, stream=True)
        valid = True
    except:
        # make sure it's an accessible file
        valid = os.path.isfile(fp)

    # if it makes it here, we're good
    return valid

def add_new_model(
    model_name,
    config_file,
    model_file=False,
    model_quant_file=False
):
    # get list of all available model configs
    model_configs = get_configs()

    assert model_name, f'Model name cannot be empty!'
    assert config_file.endswith('.yaml'), f'Model config must be .yaml, got {config_file}'

    empanada_dir = os.path.join(os.path.expanduser('~'), '.empanada')
    config_dir = os.path.join(empanada_dir, 'configs')

    # makes both dirs if needed
    os.makedirs(config_dir, exist_ok=True)
    if model_name in list(model_configs.keys()):
        print('Model name already exists!')
        model_name = model_name + 'New'
        print(f'Renaming to: {model_name}')

    # load the config file
    config = read_yaml(config_file)

    # validate the given model files
    if model_file:
        assert valid_url_or_file(model_file), \
        f"{model_file} not a valid file or url!"

        # overwrite the model file
        config['model'] = model_file
    else:
        assert valid_url_or_file(config['model']), \
        f"{config['model']} not a valid file or url!"

    if model_quant_file:
        assert valid_url_or_file(model_quant_file), \
        "{model_quant_file} not a valid file or url!"

        # overwrite the model file
        config['model_quantized'] = model_quant_file
    elif config['model_quantized'] is not None:
        if not valid_url_or_file(config['model_quantized']):
            print(f"{config['model_quantized']} not a valid file or url, ignoring.")

    # save the config file to .empanada
    with open(os.path.join(config_dir, f'{model_name}.yaml'), mode='w') as f:
        yaml.dump(config, f)

def enable_layer_rename_refresh(gui, viewer=None):
    r"""Keep a magicgui widget's Layer dropdowns in sync when a layer is renamed.

    napari only refreshes a magicgui ComboBox's layer choices when layers are
    inserted, removed, or reordered (see ``napari._qt.qt_main_window``); it does
    *not* do so when an existing layer's ``name`` changes. Without this, any
    dropdown showing layer names (e.g. "Labels layer") goes stale after a rename,
    until the plugin widget is closed and reopened.

    This attaches a listener to every current (and future) layer's
    ``events.name`` signal that refreshes all of ``gui``'s dropdown ("categorical")
    sub-widgets, so renamed layers show up immediately.

    Args:
        gui: A magicgui ``FunctionGui``, i.e. the object returned by an
            ``@magicgui``-decorated widget factory.
        viewer: Optional. If given, watches this viewer's layers directly
            (mainly useful for testing without a docked Qt widget). Otherwise,
            the viewer is resolved lazily via ``gui``'s own auto-injected
            ``viewer`` parameter, mirroring how napari resolves it elsewhere.

    Returns:
        The same ``gui`` object, for convenience chaining.

    """
    watched_layer_ids = set()
    watched_viewer_ids = set()

    def _refresh_choices(*_):
        for child in gui:
            if hasattr(child, 'reset_choices'):
                child.reset_choices()

    def _watch_layer(layer):
        if id(layer) not in watched_layer_ids:
            watched_layer_ids.add(id(layer))
            layer.events.name.connect(_refresh_choices)

    def _watch_viewer(v):
        if v is None or id(v) in watched_viewer_ids:
            return
        watched_viewer_ids.add(id(v))

        for layer in v.layers:
            _watch_layer(layer)

        def _on_inserted(event):
            _watch_layer(event.value)
            _refresh_choices()

        v.layers.events.inserted.connect(_on_inserted)

    if viewer is not None:
        _watch_viewer(viewer)
        return gui

    def _try_watch(*_):
        viewer_widget = getattr(gui, 'viewer', None)
        _watch_viewer(viewer_widget.value if viewer_widget is not None else None)

    gui.native_parent_changed.connect(_try_watch)
    _try_watch()
    return gui

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def to_tensor(img):
    # move channel dim from last to first
    tensor = torch.from_numpy(img[None])
    return tensor

class Preprocessor:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image=None):
        assert image is not None
        if np.issubdtype(image.dtype, np.floating):
            max_value = 255.0
            if image.max() <= 1.0:
                image = image * max_value
        else:
            max_value = np.iinfo(image.dtype).max

        image = normalize(image, self.mean, self.std, max_pixel_value=max_value)
        return {'image': to_tensor(image)}

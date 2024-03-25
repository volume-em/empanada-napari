import os, sys, yaml
import numpy as np
import requests
import torch
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
from empanada.config_loaders import read_yaml

__all__ = [
    'abspath'
    'get_configs',
    'Preprocessor'
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
            hash_prefix = None
            torch.hub.download_url_to_file(fpath_or_url, cached_file, hash_prefix, progress=True)

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
            raise Exception('Input image cannot be float type!')

        max_value = np.iinfo(image.dtype).max
        image = normalize(image, self.mean, self.std, max_pixel_value=max_value)
        return {'image': to_tensor(image)}

import os
import yaml

__all__ = [
    'read_yaml',
    'load_config'
]

def read_yaml(url):
    r"""
    Loads a yaml config file from the given path/url.
    """
    with open(url, mode='r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    return config

def merge_dicts(dict1, dict2):
    r"""
    Recursively merges dictionaries.
    """
    # recursive loop through keys in dict2
    for k,v in dict2.items():
        if isinstance(v, dict) and k in dict1:
            # inplace operation
            merge_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = v

    return dict1

def load_config(config_file, base_kw='BASE'):
    r"""
    Loads a config file with inheritance. The base_kw is the
    keyword that defines the parent .yaml file.
    """
    config = read_yaml(config_file)
    has_base = base_kw in config

    if not has_base:
        return config

    base_configs = [config]

    # load configs all the way to root
    while has_base:
        # get the location of the base file
        config_file_dir = os.path.dirname(config_file)
        base_path = config[base_kw]

        # load the base config
        base_config_path = os.path.join(os.path.abspath(config_file_dir), base_path)
        base_config_file = os.path.relpath(base_config_path)
        base_config = read_yaml(base_config_file)

        base_configs.append(base_config)
        has_base = base_kw in base_config
        config_file = base_config_file
        config = base_config

    # reverse the order of the configs so that the root
    # config file is first in the list
    base_configs = base_configs[::-1]

    # update keys with children overwriting parents
    inherited_config = base_configs[0]
    for config in base_configs[1:]:
        inherited_config = merge_dicts(inherited_config, config)

    return inherited_config

import os
import yaml
from attrdict import AttrDict
import torch

from lib.constants import CONFIG_ROOT

def show_gpu_info():
    """Show GPU info."""
    pass

def get_device():
    """Get best available device."""
    # return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def listdir_nohidden(path):
    fnames = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            fnames.append(f)
    return fnames

def read_yaml(path):
    """Read yaml file to AttrDict."""
    with open(path, 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.CLoader)
    return AttrDict(yaml_dict) if yaml_dict is not None else AttrDict()


def get_configs(config_name):
    default_config_path = os.path.join(CONFIG_ROOT, 'default_config.yml')
    configs = read_yaml(default_config_path)

    experiment_config_path = os.path.join(CONFIG_ROOT, config_name, 'config.yml')
    if os.path.isfile(experiment_config_path):
        configs += read_yaml(experiment_config_path)
    return configs

import os
import yaml
import pickle
from attrdict import AttrDict
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torchvision.transforms.functional import normalize, to_tensor

from lib.constants import TV_MEAN, TV_STD
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

def read_yaml_and_pickle(yaml_path):
    pickle_path = yaml_path + '.pickle'

    # if os.path.exists(pickle_path):
    #     os.remove(pickle_path)
    if os.path.exists(pickle_path) and os.stat(pickle_path).st_mtime > os.stat(yaml_path).st_mtime:
        # Read from pickle if it exists already, and has a more recent timestamp than the YAML file (no recent YAML mods)
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Reading & converting YAML to pickle: {}...".format(yaml_path))
        # Read YAML
        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.CLoader)
        # Save as pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print("Done saving to pickle.")

    return data

def read_yaml_as_attrdict(path):
    """Read yaml file to AttrDict."""
    with open(path, 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.CLoader)
    return AttrDict(yaml_dict) if yaml_dict is not None else AttrDict()


def get_configs(config_name):
    default_config_path = os.path.join(CONFIG_ROOT, 'default_config.yml')
    configs = read_yaml_as_attrdict(default_config_path)

    experiment_config_path = os.path.join(CONFIG_ROOT, config_name, 'config.yml')
    if os.path.isfile(experiment_config_path):
        configs += read_yaml_as_attrdict(experiment_config_path)
    return configs

def pillow_to_pt(image, normalize_flag=True, transform=None):
    """Pillow image to pytorch tensor."""
    if transform is not None:
        image = transform(image)
    image = np.array(image)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        # image._unsqueeze(0)
    image = to_tensor(image)
    if normalize_flag:
        image = normalize(image, TV_MEAN, TV_STD)
    return image

def pflat(x):
    a, n = x.shape
    alpha = x[np.newaxis,-1,:]
    return x / alpha

def pextend(x):
    sx, sy = x.shape
    return np.concatenate([x, np.ones((1,sy))], axis=0)

def uniform_sampling_on_S2():
    phi = np.random.uniform(low=0., high=2.*np.pi)
    theta = np.arccos(np.random.uniform(low=-1., high=1.))
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.array([x,y,z])

def get_rotation_axis_angle(axis, angle):
    T = np.eye(4)
    axis = axis / np.linalg.norm(axis)
    rotvec = angle * axis
    T[0:3, 0:3] = Rotation.from_rotvec(rotvec).as_dcm()
    return T

def get_translation(translation_vec):
    T = np.eye(4)
    T[0:3, 3] = translation_vec
    return T

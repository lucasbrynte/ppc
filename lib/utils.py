import os
import ruamel.yaml as yaml
import pickle
from attrdict import AttrDict
from collections import OrderedDict
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import uniform, norm
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from lib.constants import TV_MEAN, TV_STD

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


def read_attrdict_from_default_and_specific_yaml(default_config_path, specific_config_path):
    # Read default
    configs = read_yaml_as_attrdict(default_config_path)

    # Overwrite with specifics
    if os.path.isfile(specific_config_path):
        configs += read_yaml_as_attrdict(specific_config_path)
    return configs

def order_dict(d):
    return OrderedDict((key, d[key]) for key in sorted(d.keys()))

def numpy_to_pt(image, normalize_flag=False):
    """Numpy array to pytorch tensor."""
    if normalize_flag:
        assert not isinstance(image, np.integer)
    if len(image.shape) == 2:
        image = image[:, :, None]
    image = torch.from_numpy(image.transpose(2,0,1))
    if normalize_flag:
        image = normalize(image, TV_MEAN, TV_STD)
    return image

def get_module_parameters(module):
    w_params = []
    b_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('weight'):
            w_params.append(param)
        elif name.endswith('bias'):
            b_params.append(param)
    return w_params, b_params

def pflat(x):
    a, n = x.shape
    alpha = x[np.newaxis,-1,:]
    return x / alpha

def pextend(x):
    sx, sy = x.shape
    return np.concatenate([x, np.ones((1,sy))], axis=0)

def closeto_within(data, low=None, high=None):
    data = np.array(data)
    ret = np.ones_like(data, dtype=bool)
    if low is not None:
        undershoot = np.maximum(low - data, 0.0)
        ret &= np.isclose(undershoot, 0.0)
    if high is not None:
        overshoot = np.maximum(data - high, 0.0)
        ret &= np.isclose(overshoot, 0.0)
    return ret

def project_pts(pts, K, R, t):
    assert pts.shape[0] == 3
    return pflat(K @ (R@pts + t))[:2,0]

def uniform_sampling_on_S2(shape=()):
    """
    For shape (s1, s2, ...), output shape is (s1, s2, ..., 3)
    """
    phi = np.random.uniform(low=0., high=2.*np.pi, size=shape)
    theta = np.arccos(np.random.uniform(low=-1., high=1., size=shape))
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.moveaxis(np.array([x,y,z]), 0, -1)

def closest_rotmat(A):
    """
    Finds the rotation matrix which is "closest" to the given linear transformation.
    Returns UV^T, where USV^T is the SVD of A.
    det(A) is assumed to be > 0.
    """
    assert np.linalg.det(A) > 0
    U,_,VT = np.linalg.svd(A)
    return U @ VT

def get_rotation_axis_angle(axis, angle):
    T = np.eye(4)
    axis = axis / np.linalg.norm(axis)
    rotvec = angle * axis
    T[0:3, 0:3] = Rotation.from_rotvec(rotvec).as_dcm()
    return T

def get_translation(translation_vec):
    translation_vec = np.array(translation_vec).squeeze()
    T = np.eye(4)
    T[0:3, 3] = translation_vec
    return T

def get_2d_transl_projectivity(delta_x, delta_y):
    T = np.eye(3)
    T[0,2] = delta_x
    T[1,2] = delta_y
    return T

def get_2d_scale_projectivity(scale_x, scale_y):
    T = np.diag([scale_x, scale_y, 1.0])
    return T

def get_projectivity_for_crop_and_rescale_numpy(xc, yc, width, height, crop_dims):
    """
    When cropping and rescaling the image, the calibration matrix will also
    be affected, mapping K -> H*K, where H is the projectivity, determined
    and returned by this method.
    
    (xc, yc) - Center of bounding box. Coordinates represent pixel centers.
    (width, height) - Distance between where left-most/right-most and top-most/bottom-most pixels will be samples. Coordinates represent pixel centers.
    crop_dims - (height, width) of the resulting bounding box, in the sense of number of pixels.
    """
    desired_height, desired_width = crop_dims

    x1 = xc - 0.5*width
    y1 = yc - 0.5*height

    # Translate to map origin
    delta_x = -x1
    delta_y = -y1

    # Rescale (during which origin is fixed). Subtracting 1 from the number of pixels, assuming that pixel coordinates represent pixel centers, and that edge pixels will remain pixels during resize.
    scale_x = (desired_width - 1) / width
    scale_y = (desired_height - 1) / height

    T_transl = get_2d_transl_projectivity(delta_x, delta_y)
    T_scale = get_2d_scale_projectivity(scale_x, scale_y)
    return T_scale @ T_transl

def square_bbox_around_projected_object_center_numpy(t, K, obj_diameter, crop_box_resize_factor = 1.0):
    assert K.shape == (3, 3)
    assert t.shape == (3, 1)
    crop_dims = np.ones((2,))
    crop_dims /= t[2,0]
    crop_dims *= obj_diameter
    crop_dims *= np.diag(K)[[1,0]] / K[2,2] # Extract focal lengths fx & fy, in fy/fx order.
    crop_dims *= crop_box_resize_factor # Could expand crop_box slightly, in order to cover object even when projection is very oblique. Seems unnecessary however.
    xc, yc = pflat(K @ t)[:2,:].squeeze(1)
    height, width = crop_dims
    return xc, yc, width, height

def square_bbox_around_projected_object_center_pt_batched(t, K, obj_diameter, crop_box_resize_factor = 1.0):
    device = t.device
    xc = []
    yc = []
    width = []
    height = []
    for sample_idx in range(t.shape[0]):
        curr_xc, curr_yc, curr_width, curr_height = square_bbox_around_projected_object_center_numpy(t[sample_idx].cpu().numpy(), K[sample_idx].cpu().numpy(), obj_diameter[sample_idx].cpu().numpy(), crop_box_resize_factor = crop_box_resize_factor)
        xc.append(torch.tensor(curr_xc))
        yc.append(torch.tensor(curr_yc))
        width.append(torch.tensor(curr_width))
        height.append(torch.tensor(curr_height))
    xc = torch.stack(xc, dim=0).to(device)
    yc = torch.stack(yc, dim=0).to(device)
    width = torch.stack(width, dim=0).to(device)
    height = torch.stack(height, dim=0).to(device)
    return xc, yc, width, height

def get_2d_transl_projectivity_pt_batched(delta_x, delta_y):
    bs = delta_x.shape[0]
    device = delta_x.device
    assert delta_x.shape == (bs,)
    assert delta_y.shape == (bs,)
    T = torch.eye(3, device=device)[None,:,:].repeat(bs,1,1)
    T[:,0,2] = delta_x
    T[:,1,2] = delta_y
    return T

def get_2d_scale_projectivity_pt_batched(scale_x, scale_y):
    bs = scale_x.shape[0]
    device = scale_x.device
    assert scale_x.shape == (bs,)
    assert scale_y.shape == (bs,)
    T = torch.eye(3, device=device)[None,:,:].repeat(bs,1,1)
    T[:,0,0] = scale_x
    T[:,1,1] = scale_y
    return T

def get_projectivity_for_crop_and_rescale_pt_batched(xc, yc, width, height, crop_dims):
    """
    When cropping and rescaling the image, the calibration matrix will also
    be affected, mapping K -> H*K, where H is the projectivity, determined
    and returned by this method.
    
    (xc, yc) - Center of bounding box. Coordinates represent pixel centers.
    (width, height) - Distance between where left-most/right-most and top-most/bottom-most pixels will be samples. Coordinates represent pixel centers.
    crop_dims - (height, width) of the resulting bounding box, in the sense of number of pixels.
    """
    bs = xc.shape[0]
    assert xc.shape == (bs,)
    assert yc.shape == (bs,)
    assert width.shape == (bs,)
    assert height.shape == (bs,)

    desired_height, desired_width = crop_dims

    x1 = xc - 0.5*width
    y1 = yc - 0.5*height

    # Translate to map origin
    delta_x = -x1
    delta_y = -y1

    # Rescale (during which origin is fixed). Subtracting 1 from the number of pixels, assuming that pixel coordinates represent pixel centers, and that edge pixels will remain pixels during resize.
    scale_x = (desired_width - 1) / width
    scale_y = (desired_height - 1) / height

    T_transl = get_2d_transl_projectivity_pt_batched(delta_x, delta_y)
    T_scale = get_2d_scale_projectivity_pt_batched(scale_x, scale_y)
    return torch.matmul(T_scale, T_transl)

def crop_and_rescale_pt_batched(full_img, xc, yc, width, height, crop_dims):
    assert len(full_img.shape) == 4
    device = full_img.device
    bs = full_img.shape[0]
    assert xc.shape == (bs,)
    assert yc.shape == (bs,)
    assert width.shape == (bs,)
    assert height.shape == (bs,)
    full_img_height, full_img_width = full_img.shape[-2:]
    H = get_projectivity_for_crop_and_rescale_pt_batched(xc, yc, width, height, crop_dims)

    # Rescale H such that input & output coordinate systems are in the [-1, 1] ranges (with endpoints at centers of extreme pixels)
    H_norm2full_img = torch.matmul(
        torch.tensor(get_2d_scale_projectivity(0.5*full_img_width, 0.5*full_img_height), dtype=torch.float32), # 2nd: scale to align (2,2) (center of bottom-right corner) with correct pixel value
        torch.tensor(get_2d_transl_projectivity(1.0, 1.0), dtype=torch.float32), # 1st: translate (-1,-1) (center of top-left corner) to origin
    )[None,:,:].to(device).repeat(bs,1,1)
    H_crop_img2norm = torch.matmul(
        torch.tensor(get_2d_transl_projectivity(-1.0, -1.0), dtype=torch.float32), # 2nd: translate crop box center to origin
        torch.tensor(get_2d_scale_projectivity(1.0/(0.5*crop_dims[1]), 1.0/(0.5*crop_dims[0])), dtype=torch.float32), # 1st: scale to move (center of) bottom-right corner of crop box towards (2,2)
    )[None,:,:].to(device).repeat(bs,1,1)
    H_normalized_coords = torch.inverse(torch.matmul(H_crop_img2norm, torch.matmul(H, H_norm2full_img)))

    grid = F.affine_grid(H_normalized_coords[:,:2,:], (bs,3,crop_dims[0],crop_dims[1]), align_corners=True)
    img = F.grid_sample(full_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return H, img

def crop_img(img, crop_box, pad_if_outside=False):
    (x1, y1, x2, y2) = crop_box
    assert x2 > x1
    assert y2 > y1
    img_height, img_width = img.shape[0], img.shape[1]
    if pad_if_outside:
        xpad1 = -x1 if x1 < 0 else 0
        xpad2 = x2-img_width if x2 > img_width else 0
        ypad1 = -y1 if y1 < 0 else 0
        ypad2 = y2-img_height if y2 > img_height else 0
        # Pad image
        if len(img.shape) == 2:
            img = np.pad(img, ((ypad1, ypad2), (xpad1, xpad2)))
        else:
            assert len(img.shape) == 3
            img = np.pad(img, ((ypad1, ypad2), (xpad1, xpad2), (0, 0)))
        # If padding on negative side, compensate by shifting crop box
        x1, x2 = x1+xpad1, x2+xpad1
        y1, y2 = y1+ypad1, y2+ypad1
    else:
        assert y1 >= 0 and y2 <= img_height
        assert x1 >= 0 and x2 <= img_width
    if len(img.shape) == 2:
        img = img[y1:y2, x1:x2]
    else:
        assert len(img.shape) == 3
        img = img[y1:y2, x1:x2, :]
    return img

# def get_eucl(R, t):
#     t = np.array(t).squeeze()
#     T = np.eye(4)
#     T[0:3, 0:3] = R
#     T[0:3, 3] = t
#     return T

def sample_param(sample_spec):
    shape = sample_spec.shape if hasattr(sample_spec, 'shape') else ()
    if sample_spec.method == 'fixed':
        param = np.array(sample_spec.value)
    elif sample_spec.method == 'uniform':
        param = np.random.uniform(low=sample_spec.range[0], high=sample_spec.range[1], size=shape)
    elif sample_spec.method == 'uniform_S2':
        param = uniform_sampling_on_S2(shape=())
    elif sample_spec.method == 'normal':
        param = np.random.normal(loc=sample_spec.loc, scale=sample_spec.scale, size=shape)
    elif sample_spec.method == 'lognormal':
        param = np.exp(np.random.normal(loc=np.log(sample_spec.loc), scale=np.log(sample_spec.scale), size=shape))
    else:
        assert False, 'Unrecognized sampling method: {}'.format(sample_spec.method)
    assert param.shape == shape
    return param

def calc_param_quantile_range(sample_spec, nbr_quantiles):
    if hasattr(sample_spec, 'shape'):
        assert sample_spec.shape == (), 'Quantiles may only be computed for scalar params'
    if sample_spec.method == 'uniform':
        quantile_range = np.linspace(0., 1., nbr_quantiles)
        quantile_range = uniform.ppf(quantile_range, loc=sample_spec.range[0], scale=sample_spec.range[1]-sample_spec.range[0])
    elif sample_spec.method == 'normal':
        quantile_range = np.linspace(0., 1., nbr_quantiles+2)[1:-1] # Avoid endpoints (infinite support)
        quantile_range = norm.ppf(quantile_range, loc=sample_spec.loc, scale=sample_spec.scale)
    elif sample_spec.method == 'lognormal':
        quantile_range = np.linspace(0., 1., nbr_quantiles+2)[1:-1] # Avoid endpoints (infinite support)
        quantile_range = np.exp(norm.ppf(quantile_range, loc=np.log(sample_spec.loc), scale=np.log(sample_spec.scale)))
    else:
        assert False, 'Sampling method does not support calculating quantile range: {}'.format(sample_spec.method)
    assert quantile_range.shape == (nbr_quantiles,)
    return quantile_range

def get_human_interp_maps(configs, api):
    """
    Determine how to map output features into human-interpretable quantities.
    """
    assert api in ('torch', 'numpy')
    human_interp_maps = {}
    for task_name in configs.tasks.keys():
        unit = configs.targets[configs.tasks[task_name]['target']]['unit']
        if unit == 'px':
            human_interp_maps[task_name] = lambda x: x
        elif unit == 'angle':
            human_interp_maps[task_name] = lambda x: x * 180./math.pi
        elif unit == 'cosdist':
            human_interp_maps[task_name] = lambda x: 180./math.pi * (torch.arccos(1.0-x) if api=='torch' else np.arccos(1.0-x))
        elif unit == 'log_factor':
            human_interp_maps[task_name] = lambda x: torch.exp(x) if api=='torch' else np.exp(x)
        else:
            human_interp_maps[task_name] = lambda x: x
    return human_interp_maps

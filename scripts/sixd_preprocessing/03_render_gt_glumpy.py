
"""
Render ground truth instance segmentations, correspondence maps and normal maps from pose annotations.
"""

import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

from lib.sixd_toolkit.pysixd import inout
from scripts.sixd_preprocessing.glumpy_renderer import Renderer
import ruamel.yaml as yaml
import numpy as np
import png
from PIL import Image
import shutil


def listdir_nohidden(path):
    fnames = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            fnames.append(f)
    return fnames

# DRY_RUN = True
DRY_RUN = False

# SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/lm-lmo-from-bop/v1'
SIXD_PATH = '/datasets/lm-lmo-from-bop/v1'
SUBSETS = ['all_unoccl', 'test_occl']
# SUBSETS = [subset for subset in listdir_nohidden(SIXD_PATH) if subset.startswith('train') or subset.startswith('test')]


def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)

# def read_png(filename, dtype=None, nbr_channels=3):
#     with open(filename, 'rb') as f:
#         data = png.Reader(f).read()[2]
#         if dtype is not None:
#             img = np.vstack(map(dtype, data))
#         else:
#             img = np.vstack(data)
#     shape = img.shape
#     assert shape[1] % nbr_channels == 0
#     img = np.reshape(img, (shape[0], shape[1]//nbr_channels, nbr_channels))
#     return img

def save_png(img, filename):
    shape = img.shape
    if len(shape) == 2:
        grayscale = True
    else:
        assert len(shape) == 3 and shape[2] == 3
        grayscale = False
    with open(filename, 'wb') as f:
        writer = png.Writer(
            width = shape[1],
            height = shape[0],
            bitdepth = 16,
            greyscale = grayscale, # RGB
            alpha = False, # Not RGBA
        )
        reshaped = shape if grayscale else (shape[0], shape[1]*shape[2])
        writer.write(f, np.reshape(img, reshaped))

models_info = read_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
for j, obj_id in enumerate(sorted(models_info.keys())):
    print('Loading model {}/{}...'.format(j+1, len(models_info)))
    models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))

renderer = Renderer(
    [480, 640],
)
for obj_id, model in models.items():
    renderer._preprocess_object_model(obj_id, models[obj_id])

for subset in SUBSETS:
    for seq in sorted(listdir_nohidden(os.path.join(SIXD_PATH, subset))):
        render_seg = True
        render_instance_seg = True
        render_corr = True
        render_normals = True
        render_depth = True
        render_rgb = True

        rgb_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb')
        seg_dir = os.path.join(SIXD_PATH, subset, seq, 'seg')
        instance_seg_dir = os.path.join(SIXD_PATH, subset, seq, 'instance_seg')
        corr_dir = os.path.join(SIXD_PATH, subset, seq, 'obj')
        normals_dir = os.path.join(SIXD_PATH, subset, seq, 'normals')
        depth_rendered_dir = os.path.join(SIXD_PATH, subset, seq, 'depth_rendered')
        rgb_rendered_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb_rendered')

        if not DRY_RUN:
            if render_seg:
                if os.path.exists(seg_dir):
                    shutil.rmtree(seg_dir)
                os.makedirs(seg_dir)
            if render_instance_seg:
                if os.path.exists(instance_seg_dir):
                    shutil.rmtree(instance_seg_dir)
                os.makedirs(instance_seg_dir)
            if render_corr:
                if os.path.exists(corr_dir):
                    shutil.rmtree(corr_dir)
                os.makedirs(corr_dir)
            if render_normals:
                if os.path.exists(normals_dir):
                    shutil.rmtree(normals_dir)
                os.makedirs(normals_dir)
            if render_depth:
                if os.path.exists(depth_rendered_dir):
                    shutil.rmtree(depth_rendered_dir)
                os.makedirs(depth_rendered_dir)
            if render_rgb:
                if os.path.exists(rgb_rendered_dir):
                    shutil.rmtree(rgb_rendered_dir)
                os.makedirs(rgb_rendered_dir)

        gts = read_yaml(os.path.join(SIXD_PATH, subset, seq, 'gt.yml'))
        infos = read_yaml(os.path.join(SIXD_PATH, subset, seq, 'info.yml'))

        fnames = list(sorted(listdir_nohidden(rgb_dir)))
        for j, fname in enumerate(fnames):
            img_idx = int(fname.split('.')[0])

            info = infos[img_idx]

            if (j+1) % 100 == 0:
                print("subset {}, seq {}, frame {}/{}".format(subset, seq, j+1, len(fnames)))

            obj_id_list = []
            R_list = []
            t_list = []
            model_list = []
            for gt in gts[img_idx]:
                obj_id_list.append(gt['obj_id'])
                R_list.append(np.array(gt['cam_R_m2c']).reshape((3, 3)))
                t_list.append(np.array(gt['cam_t_m2c']).reshape((3,1)))
                model_list.append(models[gt['obj_id']])

            rgb, depth, seg, instance_seg, normal_map, corr_map = renderer.render(
                np.reshape(info['cam_K'], (3, 3)),
                R_list,
                t_list,
                obj_id_list,
                ambient_weight = 0.8,
                clip_near = 100, # mm
                clip_far = 10000, # mm
            )

            if not DRY_RUN:
                if render_seg:
                    seg_path = os.path.join(seg_dir, fname)
                    Image.fromarray(seg).save(seg_path)
                if render_instance_seg:
                    instance_seg_path = os.path.join(instance_seg_dir, fname)
                    Image.fromarray(instance_seg).save(instance_seg_path)
                if render_corr:
                    corr_path = os.path.join(corr_dir, fname)
                    save_png(corr_map, corr_path)
                if render_normals:
                    normals_path = os.path.join(normals_dir, fname)
                    save_png(normal_map, normals_path)
                if render_depth:
                    depth_rendered_path = os.path.join(depth_rendered_dir, fname)
                    save_png(depth, depth_rendered_path)
                    # Image.fromarray(depth).save(depth_rendered_path)
                if render_rgb:
                    rgb_rendered_path = os.path.join(rgb_rendered_dir, fname)
                    Image.fromarray(rgb).save(rgb_rendered_path)
            # assert False

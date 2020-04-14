import sys
import os
# sys.path.append('../..')
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

import yaml
import gdist
from lib.rigidpose.sixd_toolkit.pysixd import inout
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import argparse


DRY_RUN = False
LINEMOD_FLAG = False

if LINEMOD_FLAG:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented2cc_gdists'
else:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/ycb-video2'

parser = argparse.ArgumentParser()
parser.add_argument('--obj-id', type=int, default=None)
args = parser.parse_args()

# Load models
models_info = inout.load_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
for obj_id in models_info:
    if args.obj_id is not None and obj_id != args.obj_id:
        continue
    models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))
    print("Obj {}: {} vertices, {} faces.".format(obj_id, len(models[obj_id]['pts']), len(models[obj_id]['faces'])))


# def find_nearest_neighbors_naive(ref_points, point_cloud):
#     """
#     For each reference point, find its corresponding index in the point cloud.
#     """
#     distance_matrix = cdist(ref_points, point_cloud, metric='euclidean')
#     return np.argmin(distance_matrix, axis=1)

def find_nearest_neighbors_kdtree(ref_points, kd_tree):
    """
    For each reference point, find its corresponding index in the point cloud.
    """
    dists, closest_indices = kd_tree.query(ref_points, k=1, eps=0, p=2)
    return closest_indices

# def find_closest_vtx(x, y, z, vertices):
#     assert vertices.shape[1] == 3
#     distances = np.linalg.norm(vertices - np.array([[x, y, z]]), axis=1)
#     vtx_idx = np.argmin(distances)
#     return vtx_idx

def compute_gdists_on_model(obj_id, models, models_info):
    model = models[obj_id]
    nbr_vtx = model['pts'].shape[0]
    nbr_kp = len(models_info[obj_id]['kp_x'])
    kd_tree = cKDTree(model['pts'])
    obj_gdists = {}
    for kp_idx, kp_coords in enumerate(zip(models_info[obj_id]['kp_x'], models_info[obj_id]['kp_y'], models_info[obj_id]['kp_z'])):
        # kp_vtx_idx = find_nearest_neighbors_naive(np.array([kp_coords]), model['pts'])[0]
        # print(kp_vtx_idx)
        kp_vtx_idx = find_nearest_neighbors_kdtree(np.array([kp_coords]), kd_tree)[0]
        # print(kp_vtx_idx)
        print("")
        # kp_vtx_idx = find_closest_vtx(*kp_coords, model['pts'])
        print("Obj {}, keypoint {}/{}".format(obj_id, kp_idx+1, nbr_kp))
        obj_gdists[kp_idx] = gdist.compute_gdist(
            model['pts'].astype(np.float64),
            model['faces'].astype(np.int32),
            source_indices = np.array([kp_vtx_idx], np.int32),
            #target_indices = np.array(list(range(nbr_vtx)), np.int32),
            #max_distance = 100.0,
        )
#        colors = gdist_to_kp_per_vtx[:,np.newaxis]
#        colors = 255.999*(1.0-colors/np.max(colors))
#        models[obj_id]['colors'][:,:] = colors.astype('uint8')
#        inout.save_ply(
#            '/tmp/test.ply',
#            models[obj_id]['pts'],
#            pts_colors = models[obj_id]['colors'],
#            pts_normals = models[obj_id]['normals'],
#            faces = models[obj_id]['faces'],
#        )
#        break
    return obj_gdists

def compute_gdists_on_models(models, models_info):
    gdists = {}
    for obj_id in models.keys():
        gdists[obj_id] = compute_gdists_on_model(obj_id, models, models_info)
        # break
    return gdists

gdists_path = os.path.join(SIXD_PATH, 'models', 'gdists.yml')
if os.path.exists(gdists_path):
    with open(gdists_path, 'r') as f:
        gdists = yaml.load(f, Loader=yaml.CLoader)
else:
    gdists = {}
gdists.update(compute_gdists_on_models(models, models_info))


if not DRY_RUN:
    # Store gdists as yaml
    with open(gdists_path, 'w') as f:
        yaml.dump(gdists, f, Dumper=yaml.CDumper)

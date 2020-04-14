import sys
import os
# sys.path.append('../..')
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

import shutil
import yaml
from lib.rigidpose.sixd_toolkit.pysixd import inout
from lib.utils import listdir_nohidden
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import numpy as np
import png
from PIL import Image


DRY_RUN = False
LINEMOD_FLAG = True

if LINEMOD_FLAG:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06'
else:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/ycb-video2'


# Load models
models_info = inout.load_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))
models = {}
kd_trees = {}
for obj_id in models_info:
    models[obj_id] = inout.load_ply(os.path.join(SIXD_PATH, 'models', 'obj_{:02}.ply'.format(obj_id)))
    kd_trees[obj_id] = cKDTree(models[obj_id]['pts'])
    print("Obj {}: {} vertices, {} faces.".format(obj_id, len(models[obj_id]['pts']), len(models[obj_id]['faces'])))

def find_nearest_neighbors_naive(ref_points, point_cloud):
    """
    For each reference point, find its corresponding index in the point cloud.
    """
    distance_matrix = cdist(ref_points, point_cloud, metric='euclidean')
    return np.argmin(distance_matrix, axis=1)

def find_nearest_neighbors_kdtree(ref_points, kd_tree):
    """
    For each reference point, find its corresponding index in the point cloud.
    """
    nbr_ref_pts = ref_points.shape[0]
    dists, closest_indices = kd_tree.query(ref_points, k=1, eps=0, p=2)
    max_dist = np.max(dists)
    if not max_dist < 10.:
        # Max 10 mm to closest point - otherwise something is fishy
        print("max_dist: {}".format(max_dist))
        k = 10
        top_k_idx = np.argsort(dists)[-min(k, nbr_ref_pts):]
        top_k_dists = dists[top_k_idx]
        # top_k_dists = np.sort(dists)[-min(k, nbr_ref_pts):]
        print("top_k_dists: {}".format(top_k_dists))
        top_k_pts = ref_points[top_k_idx, :]
        print("top_k_pts: {}".format(top_k_pts))
        assert False
    return closest_indices

def project_to_surface(self, obj_id):
    distances = np.linalg.norm(self.models[obj_id]['pts'] - keypoint[np.newaxis,:], axis=1)
    closest_vtx_idx = np.argmin(distances)
    # Overwrite keypoints with closest vertices:
    return self.models[obj_id]['pts'][closest_vtx_idx_list,:]

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)

def read_png(filename, dtype=None, nbr_channels=3):
    with open(filename, 'rb') as f:
        data = png.Reader(f).read()[2]
        if dtype is not None:
            img = np.vstack(map(dtype, data))
        else:
            img = np.vstack(data)
    shape = img.shape
    assert shape[1] % nbr_channels == 0
    img = np.reshape(img, (shape[0], shape[1]//nbr_channels, nbr_channels))
    return img

def save_png(img, filename, bitdepth=16):
    shape = img.shape
    with open(filename, 'wb') as f:
        writer = png.Writer(
            width = shape[1],
            height = shape[0],
            bitdepth = bitdepth,
            greyscale = False, # RGB
            alpha = False, # Not RGBA
        )
        writer.write(f, np.reshape(img, (shape[0], shape[1]*shape[2])))

SUBSETS = sorted([subset for subset in listdir_nohidden(SIXD_PATH) if subset.startswith('train') or subset.startswith('val') or subset.startswith('test')])
# SUBSETS = ['data']

for subset in SUBSETS:
    # if subset not in [
    #     'train_synthetic',
    #     #'train_unoccl',
    #     #'train_occl',
    #     #'test_occl',
    # ]:
    #     continue
    seqs = sorted(listdir_nohidden(os.path.join(SIXD_PATH, subset)))
    #if subset == 'train_unoccl':
    #    seqs = ['driller']
    #elif subset == 'train_occl':
    #    seqs = ['ape']
    #elif subset == 'test_occl':
    #    seqs = ['benchviseblue']
    for seq in seqs:
        rgb_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb')
        instance_seg_dir = os.path.join(SIXD_PATH, subset, seq, 'instance_seg')
        corr_dir = os.path.join(SIXD_PATH, subset, seq, 'obj')
        #normals_dir = os.path.join(SIXD_PATH, subset, seq, 'normals')
        vtx_idx_dir = os.path.join(SIXD_PATH, subset, seq, 'vtx_idx')

        if not DRY_RUN:
            if os.path.exists(vtx_idx_dir):
                shutil.rmtree(vtx_idx_dir)
            os.makedirs(vtx_idx_dir)

        gts = read_yaml(os.path.join(SIXD_PATH, subset, seq, 'gt.yml'))

        fnames = list(sorted(listdir_nohidden(rgb_dir)))
        for j, fname in enumerate(fnames):
            img_idx = int(fname.split('.')[0])

            # if True:
            if (j+1) % 10 == 0:
                print("subset {}, seq {}, frame {}/{}".format(subset, seq, j+1, len(fnames)))

            instance_seg_path = os.path.join(instance_seg_dir, fname)
            corr_path = os.path.join(corr_dir, fname)
            #normals_path = os.path.join(normals_dir, fname)
            vtx_idx_path = os.path.join(vtx_idx_dir, fname)

            # Read segmentation & correspondence map
            corr_map = read_png(corr_path, dtype=np.int16, nbr_channels=3).astype('float64') + 0.5
            instance_seg = np.array(Image.open(instance_seg_path))

            img_height, img_width = instance_seg.shape

            # Vertex index map
            vtx_idx_map = np.zeros((img_height, img_width), dtype='uint32')

            instance_idx = 0
            for gt in gts[img_idx]:
                instance_idx += 1

                mask = instance_seg == instance_idx
                surface_pts = corr_map[mask,:]
                nbr_pts = surface_pts.shape[0]
                if not nbr_pts > 0:
                    continue

                obj_id = gt['obj_id']
                # print(obj_id)
                nbr_kp = len(models_info[obj_id]['kp_x'])

                # Lookup closest vertices to surface points
                #vtx_idx_map[mask] = find_nearest_neighbors_naive(surface_pts, models[obj_id]['pts'])
                #print(vtx_idx_map[mask])
                vtx_idx_map[mask] = find_nearest_neighbors_kdtree(surface_pts, kd_trees[obj_id])
                #print(vtx_idx_map[mask])

            if not DRY_RUN:
                # assert vtx_idx_map.max() < 2**16
                # Image.fromarray(vtx_idx_map.astype(np.uint16)).save(vtx_idx_path)

                # Image.fromarray(vtx_idx_map.astype(np.uint32)).save(vtx_idx_path)

                # Convert 24-bit grayscale to 8-bit little-endian RGB
                assert vtx_idx_map.max() < 2**24
                vtx_idx_map_rgb = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                for j in range(3):
                    tmp = (vtx_idx_map % 2**(8*(j+1))) // 2**(8*j)
                    assert tmp.max() < 2**8
                    vtx_idx_map_rgb[:,:,j] = tmp
                # Note: could use PIL instead:
                save_png(vtx_idx_map_rgb, vtx_idx_path, bitdepth=8)

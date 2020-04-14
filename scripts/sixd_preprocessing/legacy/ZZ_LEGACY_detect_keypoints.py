"""
Detects keypoints for objects, and writes them to <<DATA_PATH>>/models/models_info.yml
Saves 3D plots to files in <<DATA_PATH>>/models/keypoint_3dplots
"""

import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

import shutil
from lib.rigidpose.sixd_toolkit.pysixd import inout
from collections import OrderedDict
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pomegranate import GeneralMixtureModel, MultivariateGaussianDistribution
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist, pdist, squareform
from abc import ABC, abstractmethod


DRY_RUN = False
LINEMOD_FLAG = False
APPEND_WITH_FPS = True


def format_frame_idx(frame_idx):
    return '{:06}.png'.format(frame_idx)

class KeypointSelector(ABC):

    def __init__(self, opts):
        self.opts = opts
        self.models = {}
        self.models_info_backup = self.read_models_info(from_backup=True)

    def read_models_info(self, from_backup=False):
        if from_backup:
            if not os.path.exists(os.path.join(self.opts['DATA_PATH'], 'models', 'models_info_backup.yml')):
                shutil.copyfile(os.path.join(self.opts['DATA_PATH'], 'models', 'models_info.yml'), os.path.join(self.opts['DATA_PATH'], 'models', 'models_info_backup.yml'))
            models_info = inout.load_yaml(os.path.join(self.opts['DATA_PATH'], 'models', 'models_info_backup.yml'))
        else:
            models_info = inout.load_yaml(os.path.join(self.opts['DATA_PATH'], 'models', 'models_info.yml'))
        return models_info

    def get_model(self, obj):
        if obj not in self.models:
            self.models[obj] = inout.load_ply(os.path.join(self.opts['DATA_PATH'], 'models', 'obj_{:02}.ply'.format(obj)))
        return self.models[obj]

    def project_to_surface(self, kp_dict):
        for obj_id, keypoints in kp_dict.items():
            closest_vtx_idx_list = []
            # Iterate over rows:
            for keypoint in keypoints:
                distances = np.linalg.norm(self.models[obj_id]['pts'] - keypoint[np.newaxis,:], axis=1)
                closest_vtx_idx = np.argmin(distances)
                closest_vtx_idx_list.append(closest_vtx_idx)
            # Overwrite keypoints with closest vertices:
            kp_dict[obj_id] = self.models[obj_id]['pts'][closest_vtx_idx_list,:]
        return kp_dict

    def find_normals(self, kp_dict):
        normals_dict = {}
        for obj_id, keypoints in kp_dict.items():
            closest_vtx_idx_list = []
            # Iterate over rows:
            for keypoint in keypoints:
                distances = np.linalg.norm(self.models[obj_id]['pts'] - keypoint[np.newaxis,:], axis=1)
                closest_vtx_idx = np.argmin(distances)
                closest_vtx_idx_list.append(closest_vtx_idx)
            normals_dict[obj_id] = self.models[obj_id]['normals'][closest_vtx_idx_list,:]
        return normals_dict

    def store_keypoints(self, kp_dict, normals_dict=None):
        models_info_new = copy.copy(self.models_info_backup)
        for obj_id, keypoints in kp_dict.items():
            xs, ys, zs = keypoints.T
            models_info_new[obj_id]['kp_x'] = list(map(float, xs))
            models_info_new[obj_id]['kp_y'] = list(map(float, ys))
            models_info_new[obj_id]['kp_z'] = list(map(float, zs))
            if normals_dict is not None:
                xs, ys, zs = normals_dict[obj_id].T
                models_info_new[obj_id]['kp_normals_x'] = list(map(float, xs))
                models_info_new[obj_id]['kp_normals_y'] = list(map(float, ys))
                models_info_new[obj_id]['kp_normals_z'] = list(map(float, zs))
        inout.save_yaml(os.path.join(self.opts['DATA_PATH'], 'models', 'models_info.yml'), models_info_new)

    def load_keypoints(self, models_info):
        kp_dict = {}
        for obj_id in models_info:
            kp_dict[obj_id] = np.array([
                models_info[obj_id]['kp_x'],
                models_info[obj_id]['kp_y'],
                models_info[obj_id]['kp_z'],
            ]).T
        return kp_dict

    @abstractmethod
    def select_keypoints(self, initial_keypoints=None):
        pass

    def plot_keypoints(self, model, keypoints, kp_colors=None, normals=None, vtx_scores=None, store_plots=False):
        if self.opts['SCORES_COLORED_IN_SCATTERPLOT']:
            assert vtx_scores is not None
        if self.opts['MIN_VTX_SCORE_SCATTERPLOT'] is not None:
            assert vtx_scores is not None

        if store_plots:
            os.makedirs(os.path.join(self.opts['DATA_PATH'], 'models', 'keypoint_3dplots'), exist_ok=True)
        # for obj_id, all_scores in vtx_scores_filtered.items():
        #     # scores = np.random.choice(all_scores)
        #     # scores = all_scores[0]
        #     scores = sum(all_scores) / float(len(all_scores))
        print("Plotting object {}".format(obj_id))

        model = self.get_model(obj_id)

        # plt.figure()
        # plt.hist(scores, bins=100)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if self.opts['MIN_VTX_SCORE_SCATTERPLOT'] is None:
            xs, ys, zs = model['pts'].T
        else:
            xs, ys, zs = model['pts'][scores >= self.opts['MIN_VTX_SCORE_SCATTERPLOT'], :].T
        if self.opts['SCORES_COLORED_IN_SCATTERPLOT']:
            if self.opts['MIN_VTX_SCORE_SCATTERPLOT'] is None:
                cvals = scores**self.opts['SCORE_EXP']
            else:
                cvals = scores[scores >= self.opts['MIN_VTX_SCORE_SCATTERPLOT']]**self.opts['SCORE_EXP']
            # cvals -= np.min(cvals)
            # cvals /= np.max(cvals)
        if self.opts['MAX_NBR_VTX_SCATTERPLOT'] is None:
            self.opts['MAX_NBR_VTX_SCATTERPLOT'] = len(xs)
        choice = np.random.choice(len(xs), self.opts['MAX_NBR_VTX_SCATTERPLOT'])
        ax.scatter(
            xs[choice],
            ys[choice],
            zs[choice],
            c=cvals[choice] if self.opts['SCORES_COLORED_IN_SCATTERPLOT'] else None,
            cmap=plt.get_cmap('Greens') if self.opts['SCORES_COLORED_IN_SCATTERPLOT'] else None,
            norm=colors.Normalize(vmin=self.opts['SCATTER_VMIN']**self.opts['SCORE_EXP'], vmax=self.opts['SCATTER_VMAX']**self.opts['SCORE_EXP'], clip=False),
            # norm=colors.Normalize(),
            # norm=colors.LogNorm(),
        )

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        if kp_colors is not None:
            for kp, color in zip(keypoints, kp_colors):
                ax.plot([kp[0]], [kp[1]], [kp[2]], '*', color=color, markersize=self.opts['MARKERSIZE'])
        else:
            ax.plot(*keypoints.T, 'r*', markersize=self.opts['MARKERSIZE'])

        if normals is not None:
            normal_length_mm = 10.0
            ax.quiver(*keypoints.T, *(normal_length_mm*normals).T)

        if store_plots:
            plt.savefig(os.path.join(self.opts['DATA_PATH'], 'models', 'keypoint_3dplots', 'obj_{:02}.png'.format(obj_id)))
        else:
            plt.show()


class DetectorKeypointSelector(KeypointSelector):

    def __init__(self, opts):
        super().__init__(opts)

        # self.detector = cv2.FeatureDetector_create("SIFT")
        # self.detector = cv2.xfeatures2d_SIFT()
        self.detector = cv2.ORB_create(nfeatures=20000)

    def _score_vertices(self):
        vtx_scores = OrderedDict()
        instance_counts = OrderedDict()

        # Loop over all sequences
        # for seq in ['01']: # ape
        # for seq in ['02']: # benchvise
        # for seq in ['06']: # cat?
        # for seq in ['09']: # duck
        # for seq in ['12']: # holepuncher
        # for seq in ['13']: # iron
        for seq in sorted(os.listdir(os.path.join(self.opts['DATA_PATH'], self.opts['TRAIN_SUBDIR']))):
            info = inout.load_info(os.path.join(self.opts['DATA_PATH'], self.opts['TRAIN_SUBDIR'], seq, 'info.yml'))
            gt = inout.load_gt(os.path.join(self.opts['DATA_PATH'], self.opts['TRAIN_SUBDIR'], seq, 'gt.yml'))
            assert len(info) == len(gt)
            nbr_frames = len(info)

            # Loop over all images
            if self.opts['NBR_FRAMES_SAMPLED_PER_SEQ'] is None:
                frames = list(range(nbr_frames))
            else:
                frames = sorted(np.random.choice(nbr_frames, self.opts['NBR_FRAMES_SAMPLED_PER_SEQ']))
            for frame_idx in frames:
                print("Seq: {}, frame: {}, objects: {}".format(seq, frame_idx, list(map(lambda x: x['obj_id'], gt[frame_idx]))))
                K = info[frame_idx]['cam_K']
                # Unnecessary:
                # R_w2c = info['cam_R_w2c'] if 'cam_R_w2c' in info else np.eye(3)
                # t_w2c = info['cam_t_w2c'] if 'cam_t_w2c' in info else np.zeros((3,1))
                # info['depth_scale'] also unnecessary, no need to read/scale depth images

                img = cv2.imread(os.path.join(self.opts['DATA_PATH'], self.opts['TRAIN_SUBDIR'], seq, 'rgb', format_frame_idx(frame_idx)))

                # NOTE: Rendering segmentations requires either one of:
                #           Modifying C++ renderer to read BOP annotations
                #           Modify BOP python renderer's shaders to produce seg (hopefully not too hard to reuse shader code from C++ renderer)
                # seg = cv2.imread(os.path.join(self.opts['DATA_PATH'], self.opts['TRAIN_SUBDIR'], seq, 'seg', format_frame_idx(frame_idx)))

                # NOTE: Detector applied on RGB (or is it BGR?) image, i.e. not grayscale. Not sure what the implications of this are.
                all_keypoints = self.detector.detect(img)
                print("Found {} feature points (keypoint candidates)".format(len(all_keypoints)))

                # print("Total #keypoints: {}".format(len(all_keypoints)))

                # Loop over all object instances
                for instance in gt[frame_idx]:
                    if instance['obj_id'] not in self.models_info_backup.keys():
                        print("Discarding object: {}'.format(instance['obj_id'])'")
                        continue

                    if instance['obj_id'] in instance_counts:
                        instance_counts[instance['obj_id']] += 1
                    else:
                        instance_counts[instance['obj_id']] = 1

                    model = self.get_model(instance['obj_id'])
                    R_m2c = instance['cam_R_m2c']
                    t_m2c = instance['cam_t_m2c']
                    bbox = instance['obj_bb'] # xmin, ymin, width, height
                    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                    # print("Object: {}".format(instance['obj_id']))
                    # print(xmin, xmax, ymin, ymax)
                    # plt.figure()
                    # plt.imshow(img)
                    # plt.show()

                    # Determine which keypoints belong to current object instance
                    # OpenCV coordinates correspond to pixel centers according to http://answers.opencv.org/question/35111/origin-pixel-in-the-image-coordinate-system-in-opencv/
                    keypoints = []
                    for kp in all_keypoints:
                        x = int(0.5+kp.pt[0])
                        y = int(0.5+kp.pt[1])

                        # If seg would be used:
                        # if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                        #     continue
                        # if seg[y,x] != instance['obj_id']:
                        #     continue

                        # NOTE: Using bbox eliminates need for rendering segmentations. Some keypoints might be outside object, but hopefully these effects are negligible with statistics from enough frames.
                        if x < xmin or x > xmax or y < ymin or y > ymax:
                            # print("Projects outside bounding box (does not belong to object):")
                            # print(x, xmin, xmax)
                            # print(y, ymin, ymax)
                            continue
                        keypoints.append(kp)

                    if len(keypoints) == 0:
                        print("No keypoints found for object {}!".format(instance['obj_id']))
                        continue

                    sigma = np.array(list(map(lambda x: x.size, keypoints)))
                    # Fairly confident order is correct - 1st row is horizontal coordinate, 2nd row is vertical
                    x_2d = np.array(list(map(lambda x: x.pt, keypoints))).T
                    detection_score = np.array(list(map(lambda x: x.response, keypoints)))

                    # Find top-k keypoints based on response
                    score_sorted = np.sort(detection_score)
                    strong_kp_mask = detection_score >= score_sorted[max(0, len(score_sorted)-self.opts['MAX_NBR_FEATURES_DETECTED'])]

                    # Select only these top-k keypoints
                    sigma = sigma[strong_kp_mask]
                    sigma *= self.opts['FEATURE_SCALE_FACTOR']
                    x_2d = x_2d[:, strong_kp_mask]
                    detection_score = detection_score[strong_kp_mask]

                    nbr_kp = x_2d.shape[1]

                    # Homogeneous coordinates
                    x = np.concatenate(
                        (x_2d, np.ones((1, nbr_kp))),
                        axis=0,
                    )

                    # Normalize pixel coordinates using calibration
                    # 3D points in image plane, camera coordinate system
                    X_imgplane_cam = np.linalg.solve(K, x)
                    # Transform to model coordinate system & normalize to get viewing rays.
                    X_imgplane = R_m2c.T @ (X_imgplane_cam - t_m2c)
                    # Shape: (3, nbr_kp)
                    vrays = X_imgplane / np.linalg.norm(X_imgplane, axis=0)

                    # Project vertices to 2 components, one parallel and one orthogonal to viewing ray.
                    # Shape: (3, nbr_vtx)
                    all_vtx = model['pts'].T
                    nbr_vtx = all_vtx.shape[1]

                    # Dot product along axis 0 of arrays with shape: (3, nbr_kp, nbr_vtx)
                    # Shape: (nbr_kp, nbr_vtx)
                    parallel_coordinates = np.sum(vrays[:,:,np.newaxis]*all_vtx[:,np.newaxis,:], axis=0)

                    # Filter vertices based on depth
                    # Shape: (nbr_kp, nbr_vtx)
                    small_depth_vtx_mask = parallel_coordinates >= np.min(parallel_coordinates, axis=1, keepdims=True) + self.opts['DEPTH_DIFF_TH']

                    # Non-vectorized implementation.
                    # if instance['obj_id'] not in vtx_scores:
                    #     vtx_scores[instance['obj_id']] = np.zeros((nbr_vtx,))
                    # for kp_idx in range(nbr_kp):
                    #     print(kp_idx)
                    #     for vtx_idx in range(nbr_vtx):
                    #         if small_depth_vtx_mask[kp_idx, vtx_idx]:
                    #             dist_to_ray = np.linalg.norm(all_vtx[:,vtx_idx] - parallel_coordinates[kp_idx, vtx_idx]*vrays[:,kp_idx])
                    #             # TODO: Maybe abort iteration if distance big?
                    #             dist_weight = np.exp(-dist_to_ray**2 / (2.0*(0.5*(K[0,0]+K[1,1])*sigma[kp_idx])**2))
                    #             vtx_scores[instance['obj_id']][vtx_idx] += detection_score[kp_idx] * dist_weight

                    # Shape: (3, nbr_kp, nbr_vtx)
                    # all_vtx_parallel = parallel_coordinates[np.newaxis,:,:] * vrays[:,:,np.newaxis]
                    # all_vtx_orthogonal = all_vtx[:,np.newaxis,:] - all_vtx_parallel
                    # 
                    # # Shape: (nbr_kp, nbr_vtx)
                    # dists = np.linalg.norm(all_vtx_orthogonal, axis=0)
                    
                    all_vtx_cam_frame = R_m2c @ all_vtx + t_m2c
                    all_vtx_proj = K @ all_vtx_cam_frame
                    all_vtx_proj_pixels = all_vtx_proj[0:2,:] / all_vtx_proj[np.newaxis,2,:]
                    dists = np.linalg.norm(all_vtx_proj_pixels[:,np.newaxis,:] - x_2d[:,:,np.newaxis], axis=0)

                    # Shape: (nbr_kp, nbr_vtx)
                    dist_weight = np.exp(-dists**2 / (2.0*(sigma[:,np.newaxis])**2))

                    # Shape: (nbr_vtx,)
                    if self.opts['DIFFERENTIATE_ON_KP_RESPONSE']:
                        scores = np.sum(small_depth_vtx_mask * dist_weight * detection_score[:,np.newaxis], axis=0)
                    else:
                        scores = np.sum(small_depth_vtx_mask * dist_weight, axis=0)

                    if instance['obj_id'] in vtx_scores:
                        vtx_scores[instance['obj_id']].append(scores)
                    else:
                        vtx_scores[instance['obj_id']] = [scores]

                    # if instance['obj_id'] in vtx_scores:
                    #     k = instance_counts[instance['obj_id']]
                    #     old_scores = vtx_scores[instance['obj_id']]
                    #     vtx_scores[instance['obj_id']] = (k-1.0)/k*old_scores + 1.0/k*scores
                    # else:
                    #     vtx_scores[instance['obj_id']] = scores

                    # break #instance
                # if frame_idx > 10:
                #     break
            #     break #frame
            # break #seq
        return vtx_scores

    def _smooth_vertex_scores(self, vtx_scores):
        vtx_scores_filtered = OrderedDict()
        for obj_id, all_scores in vtx_scores.items():
            scores_raw = sum(all_scores) / float(len(all_scores))
            model = self.get_model(obj_id)

            nbr_vtx = model['pts'].shape[0]
            vtx_subset = np.random.choice(range(nbr_vtx), self.opts['LP_DISTMAT_SUBSET_SIZE'])
            # distance_matrix = squareform(pdist(model['pts'], metric='euclidean'))
            distance_matrix = cdist(model['pts'][vtx_subset], model['pts'], metric='euclidean')
            kernel = np.exp(-0.5*(distance_matrix / self.opts['LP_SIGMA_MM'])**2)
            scores_lowpass = np.sum(scores_raw[vtx_subset, np.newaxis] * kernel, axis=0) / np.sum(kernel, axis=0)

            # scores = scores_raw - scores_lowpass
            scores = scores_raw / scores_lowpass
            # scores = scores_lowpass

            print(np.min(scores))
            print(np.max(scores))

            vtx_scores_filtered[obj_id] = scores
            # vtx_scores_filtered[obj_id] = [scores]
        return vtx_scores_filtered

    def _get_keypoints_from_vtx_scores_via_gmm_fitting(self, vtx_scores_filtered):
        # for obj_id, all_scores in vtx_scores_filtered.items():
        #     # scores = np.random.choice(all_scores)
        #     # scores = all_scores[0]
        #     scores = sum(all_scores) / float(len(all_scores))
        kp_dict = {}
        for obj_id, scores in vtx_scores_filtered.items():
            print("Inferring GMM for object {}".format(obj_id))

            model = self.get_model(obj_id)

            # X = X[np.random.choice(X.shape[0], 1000),:]
            if self.opts['MIN_VTX_SCORE_GMM'] is not None:
                X = model['pts'][scores >= self.opts['MIN_VTX_SCORE_GMM'], :]
                weights = scores[scores >= self.opts['MIN_VTX_SCORE_GMM']]**self.opts['SCORE_EXP']
            else:
                X = model['pts']
                weights = scores**self.opts['SCORE_EXP']

            gmm_model = GeneralMixtureModel.from_samples(
                MultivariateGaussianDistribution,
                n_components=self.opts['NBR_KEYPOINTS'],
                init='kmeans++',
                # init='random',
                X=X,
                weights=weights,
            )

            keypoints = np.array([d.mu for d in gmm_model.distributions])

            if self.opts['MAX_DIST_MM_FROM_KP_TO_SURFACE'] is not None:
                distance_matrix_kp_to_vtx = cdist(keypoints, model['pts'], metric='euclidean')
                kp_distances_to_surface = np.min(distance_matrix_kp_to_vtx, axis=1)
                mask = kp_distances_to_surface <= self.opts['MAX_DIST_MM_FROM_KP_TO_SURFACE']**2
                keypoints = keypoints[mask, :]

            kp_dict[obj_id] = keypoints

        return kp_dict

    def select_keypoints(self, initial_keypoints=None):
        assert initial_keypoints is None
        vtx_scores = self._score_vertices()
        vtx_scores_filtered = self._smooth_vertex_scores(vtx_scores)
        kp_dict = self._get_keypoints_from_vtx_scores_via_gmm_fitting(vtx_scores_filtered)
        return kp_dict


class FarthestPointSamplingKeypointSelector(KeypointSelector):

    def __init__(self, opts):
        super().__init__(opts)

    def _get_farthest_point(self, curr_pts, all_pts):
        distance_matrix = cdist(curr_pts, all_pts, metric='euclidean')

        # Distance from each vertex to its closest point among curr_pts
        all_pts_dist_to_closest = np.min(distance_matrix, axis=0)

        return all_pts[np.argmax(all_pts_dist_to_closest), :]

    def initialize_keypoints(self):
        kp_dict = {}
        for obj_id in self.models_info_backup.keys():
            model = self.get_model(obj_id)
            keypoints = np.empty((0,3))
            origin = np.zeros((3,)) # Model center
            initial_kp = self._get_farthest_point(origin[np.newaxis,:], model['pts'])
            keypoints = np.concatenate((keypoints, initial_kp[np.newaxis,:]), axis=0)
            kp_dict[obj_id] = keypoints
        return kp_dict

    def select_keypoints(self, initial_keypoints=None):
        if initial_keypoints is None:
            kp_dict = self.initialize_keypoints()
        else:
            kp_dict = copy.copy(initial_keypoints)
        for obj_id in self.models_info_backup.keys():
            model = self.get_model(obj_id)
            initial_nbr_kp = kp_dict[obj_id].shape[0]
            for _ in range(initial_nbr_kp, self.opts['NBR_KEYPOINTS']):
                new_kp = self._get_farthest_point(kp_dict[obj_id], model['pts'])
                kp_dict[obj_id] = np.concatenate((kp_dict[obj_id], new_kp[np.newaxis,:]), axis=0)
        return kp_dict

if LINEMOD_FLAG:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06'
    SUBSET = 'train_unoccl'
else:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/ycb-video2'
    SUBSET = 'train'

if not DRY_RUN:
    STORE_KEYPOINTS = True
    STORE_PLOTS = True
else:
    STORE_KEYPOINTS = False
    STORE_PLOTS = False

PROJECT_TO_SURFACE = True # Replace keypoints with closest vertices
FIND_NORMALS = True # Assuming keypoints close to surface. Evaluates normal at closest vertex.



opts = {
    # 'MARKERSIZE': 10,
    'MARKERSIZE': 30,
    'MAX_NBR_VTX_SCATTERPLOT': 500,
    'NBR_KEYPOINTS': 10,
    'SCORES_COLORED_IN_SCATTERPLOT': False,
    'MIN_VTX_SCORE_GMM': None,
    'MIN_VTX_SCORE_SCATTERPLOT': None,
    'SCATTER_VMIN': 0.0,
    'SCATTER_VMAX': 10.0,
    'SCORE_EXP': 1.0,
    'DATA_PATH': SIXD_PATH,
}
opts.update({
    'DIFFERENTIATE_ON_KP_RESPONSE': False,
    'MAX_NBR_FEATURES_DETECTED': 100,
    # 'DIST_TH': 1e-2, # meters
    'MAX_DIST_MM_FROM_KP_TO_SURFACE': 3.0,
    'FEATURE_SCALE_FACTOR': 1e-1,
    'NBR_FRAMES_SAMPLED_PER_SEQ': 100 if LINEMOD_FLAG else 10,
    'LP_SIGMA_MM': 40.0,
    'LP_DISTMAT_SUBSET_SIZE': 1000,
    'DEPTH_DIFF_TH': 1e-2, # meters
    'TRAIN_SUBDIR': SUBSET, # Images in this subdir will be used to collect keypoint statistics
})
kp_selector = DetectorKeypointSelector(opts)
initial_keypoints = kp_selector.select_keypoints(initial_keypoints=None)


if not APPEND_WITH_FPS:
    kp_dict = initial_keypoints
else:
    opts = {
        # 'MARKERSIZE': 10,
        'MARKERSIZE': 30,
        'MAX_NBR_VTX_SCATTERPLOT': 500,
        'NBR_KEYPOINTS': 20,
        'SCORES_COLORED_IN_SCATTERPLOT': False,
        'MIN_VTX_SCORE_GMM': None,
        'MIN_VTX_SCORE_SCATTERPLOT': None,
        'SCATTER_VMIN': 0.0,
        'SCATTER_VMAX': 10.0,
        'SCORE_EXP': 1.0,
        'DATA_PATH': SIXD_PATH,
    }
    kp_selector = FarthestPointSamplingKeypointSelector(opts)

    # Select features
    # kp_dict = kp_selector.select_keypoints()
    kp_dict = kp_selector.select_keypoints(initial_keypoints=initial_keypoints)
    if PROJECT_TO_SURFACE:
        kp_dict = kp_selector.project_to_surface(kp_dict)
    normals_dict = kp_selector.find_normals(kp_dict) if FIND_NORMALS else None
    if STORE_KEYPOINTS:
        kp_selector.store_keypoints(kp_dict, normals_dict=normals_dict)

# Or read features from file
# kp_dict = kp_selector.load_keypoints(kp_selector.read_models_info(from_backup=False))

# Plot selected features
for obj_id, keypoints in kp_dict.items():
    print("Found {} keypoints for object {}".format(len(keypoints), obj_id))
    kp_selector.plot_keypoints(
        kp_selector.get_model(obj_id),
        keypoints,
        kp_colors = plt.cm.tab20.colors,
        normals = normals_dict[obj_id] if FIND_NORMALS else None,
        vtx_scores = vtx_scores_filtered[obj_id] if opts['SCORES_COLORED_IN_SCATTERPLOT'] else None,
        store_plots = STORE_PLOTS,
    )

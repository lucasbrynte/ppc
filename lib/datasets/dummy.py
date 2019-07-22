"""Reading input data in the SIXD common format."""
from collections import namedtuple
import os
import shutil
import yaml

import math
import numpy as np
from PIL import Image
import torch
from torch import Tensor
# from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset

from lib.utils import read_yaml_and_pickle, pflat, pillow_to_pt
from lib.utils import uniform_sampling_on_S2, get_rotation_axis_angle, get_translation
from lib.constants import TRAIN, VAL
from lib.loader import Sample
from lib.sixd_toolkit.pysixd import inout
from lib.rendering.glumpy_renderer import Renderer
from lib.rendering.pose_sampler import PoseSampler

def get_metadata(configs):
    path = os.path.join(configs.data.path, 'models', 'models_info.yml')
    with open(path, 'r') as file:
        models_info = yaml.load(file, Loader=yaml.CLoader)
    def rows2array(obj_anno, prefix):
        return np.array([
            obj_anno[prefix + '_x'],
            obj_anno[prefix + '_y'],
            obj_anno[prefix + '_z'],
        ])
    def get_bbox3d(min_bounds, diff):
        max_bounds = min_bounds + diff
        return np.hstack([min_bounds[:,None], max_bounds[:,None]])
    return {
        'objects': {obj_anno['readable_label']: {
            'bbox3d': get_bbox3d(rows2array(obj_anno, 'min'), rows2array(obj_anno, 'size')),
            'keypoints': rows2array(obj_anno, 'kp'),
            'kp_normals': rows2array(obj_anno, 'kp_normals'),
        } for obj_label, obj_anno in models_info.items()},
    }

def get_dataset(configs, mode):
    return DummyDataset(configs, mode)


Annotation = namedtuple('Annotation', [
    'pixel_offset',
    'rel_depth_error',
    'delta_angle_inplane',
    'delta_theta',
    'delta_R33',
    'norm_pixel_offset',
    'abs_delta_angle_inplane',
    'abs_delta_theta',
])

global global_renderer
global_renderer = None

class DummyDataset(Dataset):
    def __init__(self, configs, mode):
        self._configs = configs
        self._metadata = get_metadata(configs)
        self._mode = mode
        self._K = self._read_calibration()
        self._models_info = self._init_models_info()
        self._obj_label = self._configs.obj_label
        self._obj_id = self._determine_obj_id()
        self._model = self._init_model()
        self._renderer = self._init_renderer()
        self._aug_transform = None
        # if self._mode == TRAIN:
        #     self._aug_transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03)
        #     # self._aug_transform = ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5), saturation=(0.7, 1.5), hue=(-0.03, 0.03))
        # else:
        #     self._aug_transform = None

        self._pids_path = '/tmp/sixd_kp_pids/' + self._mode
        if os.path.exists(self._pids_path):
            shutil.rmtree(self._pids_path)
            print("Removing " + self._pids_path)
        print("Creating " + self._pids_path)
        os.makedirs(self._pids_path)

    def _read_yaml(self, path):
        return read_yaml_and_pickle(path)

    def _read_calibration(self):
        calib = self._read_yaml(os.path.join(self._configs.data.path, 'camera.yml'))
        K = np.array([
            [calib['fx'],           0.0,   calib['cx']],
            [0.0,           calib['fy'],   calib['cy']],
            [0.0,                   0.0,           1.0],
        ])
        return K

    def _init_models_info(self):
        return self._read_yaml(os.path.join(self._configs.data.path, 'models', 'models_info.yml'))

    def _determine_obj_id(self):
        filtered_obj_ids = [obj_id for obj_id, model_spec in self._models_info.items() if model_spec['readable_label'] == self._obj_label]
        assert len(filtered_obj_ids) == 1
        obj_id = filtered_obj_ids[0]
        return obj_id

    def _init_model(self):
        print("Loading model...")
        model = inout.load_ply(os.path.join(self._configs.data.path, 'models', 'obj_{:02}.ply'.format(self._obj_id)))
        print("Done.")
        return model

    def _init_renderer(self):
        global global_renderer
        if global_renderer is not None:
            print('Reusing renderer')
            return global_renderer
        renderer = Renderer(
            self._configs.data.crop_dims,
        )
        renderer._preprocess_object_model(self._obj_id, self._model)
        print('Not reusing renderer')
        global_renderer = renderer
        return renderer

    def __len__(self):
        return 512

    def _init_worker_seed(self):
        pid = os.getpid()
        pid_path = os.path.join(self._pids_path, str(pid))
        if os.path.exists(pid_path):
            return
        np.random.seed(pid)
        open(pid_path, 'w').close()

    def __getitem__(self, index):
        self._init_worker_seed() # Cannot be called in constructor, since it is only executed by main process. Workaround: call at every sampling.
        # self._renderer = self._init_renderer()
        data, annotations = self._generate_sample()
        return Sample(annotations, data)

    def _render(self, K, R, t):
        rgb, depth, seg, instance_seg, normal_map, corr_map = self._renderer.render(
            K,
            [R],
            [t],
            [self._obj_id],
            ambient_weight = 0.8,
            clip_near = 100, # mm
            clip_far = 10000, # mm
        )

        return rgb

    def _truncate_bbox(self, bbox):
        (x1, y1, x2, y2) = bbox
        x1, x2 = np.clip((x1, x2), 0, self._configs.data.img_dims[1])
        y1, y2 = np.clip((y1, y2), 0, self._configs.data.img_dims[0])
        truncated_bbox = (x1, y1, x2, y2)
        return truncated_bbox

    def _shift_bbox_into_img(self, bbox):
        (x1, y1, x2, y2) = bbox
        assert x2 - x1 <= self._configs.data.img_dims[1]
        assert y2 - y1 <= self._configs.data.img_dims[0]

        if x1 < 0:
            xshift = 0 - x1
        elif x2 > self._configs.data.img_dims[1]:
            xshift = - (x2 - self._configs.data.img_dims[1])
        else:
            xshift = 0

        if y1 < 0:
            yshift = 0 - y1
        elif y2 > self._configs.data.img_dims[0]:
            yshift = - (y2 - self._configs.data.img_dims[0])
        else:
            yshift = 0

        x1, x2 = x1+xshift, x2+xshift
        y1, y2 = y1+yshift, y2+yshift

        shifted_bbox = (x1, y1, x2, y2)
        return shifted_bbox

    def _bbox_from_projected_keypoints(self, R, t):
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        keypoints_3d = self._metadata['objects'][self._obj_label]['keypoints']
        keypoints_2d = pflat(self._K @ (R @ keypoints_3d + t))[:2,:]
        x1, y1 = np.min(keypoints_2d, axis=1)
        x2, y2 = np.max(keypoints_2d, axis=1)
        bbox = (x1, y1, x2, y2)
        bbox = self._truncate_bbox(bbox)
        return bbox

    def _wrap_bbox_in_squarebox(self, bbox):
        # Only square dimensions supported as of now (one way to ensure that every bbox may be wrapped in a square):
        assert self._configs.data.img_dims[0] == self._configs.data.img_dims[1]

        (x1_old, y1_old, x2_old, y2_old) = bbox
        old_width = x2_old - x1_old
        old_height = y2_old - y1_old
        if old_width < old_height:
            new_height = old_height
            new_width  = old_height
        else:
            new_width  = old_width
            new_height = old_width
        delta_height = new_height - old_height
        delta_width = new_width - old_width
        assert delta_height >= 0
        assert delta_width >= 0
        xc = (x1_old + x2_old) * 0.5
        yc = (y1_old + y2_old) * 0.5
        x2_new = int(xc + 0.5*new_width)
        x1_new = int(x2_new - new_width)
        y2_new = int(yc + 0.5*new_height)
        y1_new = int(y2_new - new_height)
        square_bbox = (x1_new, y1_new, x2_new, y2_new)
        square_bbox = self._shift_bbox_into_img(square_bbox)
        return square_bbox

    def _get_transl_projectivity(self, delta_x, delta_y):
        T = np.eye(3)
        T[0,2] = delta_x
        T[1,2] = delta_y
        return T

    def _get_scale_projectivity(self, scale_x, scale_y):
        T = np.diag([scale_x, scale_y, 1.0])
        return T

    def _get_projectivity_for_crop_and_rescale(self, crop_box):
        """
        When cropping and rescaling the image, the calibration matrix will also
        be affected, mapping K -> T*K, where T is the projectivity, determined
        and returned by this method.
        """
        (x1, y1, x2, y2) = crop_box

        # Pixel width & height of region of interest
        old_width = x2 - x1
        old_height = y2 - y1
        new_height, new_width = self._configs.data.crop_dims

        # Translate to map origin
        delta_x = -x1
        delta_y = -y1

        # Rescale (during which origin is fixed)
        scale_x = new_width / old_width
        scale_y = new_height / old_height

        T_transl = self._get_transl_projectivity(delta_x, delta_y)
        T_scale = self._get_scale_projectivity(scale_x, scale_y)
        return T_scale @ T_transl

    def _apply_perturbation(self, T1):
        # 90 degrees
        MAX_ANGLE = np.pi / 2
        
        # 30% of object size along each dimension
        MAX_OBJECT_BIAS_MM = 0.3 * np.diff(self._metadata['objects'][self._obj_label]['bbox3d'], axis=1).squeeze()

        # 300 mm
        MAX_DEPTH_BIAS_MM = 300.

        random_axis = uniform_sampling_on_S2()
        random_angle = np.random.uniform(low=0., high=MAX_ANGLE)
        random_transl = np.random.uniform(low=-MAX_OBJECT_BIAS_MM, high=MAX_OBJECT_BIAS_MM, size=(3,))
        random_depth_bias = np.random.uniform(low=-MAX_DEPTH_BIAS_MM, high=MAX_DEPTH_BIAS_MM)

        # Note: perturbation in object frame. We want to apply rotations around object center rather than camera center (which would be quite uncontrolled).
        T_perturb_obj = get_translation(random_transl) @ get_rotation_axis_angle(random_axis, random_angle)

        # Additional stronger perturbation along principal axis
        T_perturb_depth = get_translation(np.array([0., 0., random_depth_bias]))
        T2 = T_perturb_depth @ T1 @ T_perturb_obj
        return T2

    def _sample_pose_pair(self):
        pose_sampler = PoseSampler()
        up_dir = np.array([0., 0., 1.])
        zmin = self._metadata['objects'][self._obj_label]['bbox3d'][2,0]
        bottom_center = np.array([0., 0., zmin])
        table_size = 1000. # 1000 mm
        T_model2world = pose_sampler.sample_object_pose_on_xy_plane(up_dir, bottom_center, table_size)
        T_world2cam = pose_sampler.sample_camera_pose(
            hemisphere_polar_angle_range = [0., np.pi/2],
            hemisphere_radius_range = [700., 1500.], # 700 - 1500 mm
            inplane_rot_angle_range = [-np.pi/6, np.pi/6],
            principal_axis_perturb_angle_range = [-np.pi/6, np.pi/6],
        )
        T1 = T_world2cam @ T_model2world

        # Perturb T1, to get T2
        T2 = self._apply_perturbation(T1)

        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T1[3,:], np.array([0., 0., 0., 1.])))
        assert np.all(np.isclose(T2[3,:], np.array([0., 0., 0., 1.])))

        # Extract and return rotations & translations
        R1 = T1[:3,:3]; t1 = T1[:3,[3]]
        R2 = T2[:3,:3]; t2 = T2[:3,[3]]
        return R1, t1, R2, t2

    def _calc_delta_angle_inplane(self, R):
        # NOTE: Perturbation of R33 element from 1 determines to what extent the rotation deviates from an inplane rotation
        # When R33 is close to 1, rot is close to inplane-rot, determined by the upper left 2x2 block
        MIN_R33 = 0.1
        if not R[2,2] >= MIN_R33:
            # Return max value
            return math.pi
        # Extract 2x2 block
        A = R[:2,:2]
        approx_inplane_angle = self._angle_from_2x2_approx_rotmat(A) # Determinant of subblock > 0 when R33 > 0
        return approx_inplane_angle

    def _angle_from_2x2_approx_rotmat(self, A):
        """
        Takes a 2x2 linear transformation, and computes the angle of the
        rotation UV^T, where USV^T is the SVD of A.

        This correspond to the actual angle by which the columns of V would be
        rotated by A (the scaling of S will not affect this angle).

        Another way to see it is that A maps the unit circle to an ellipse,
        where the columns of V are kept orthogonal when mapped to the semi-axes
        of the ellipse.
        """
        assert A.shape == (2, 2)
        assert np.linalg.det(A) > 0
        U,_,VT = np.linalg.svd(A)
        return self._angle_from_rotmat(U @ VT)

    def _angle_from_rotmat(self, R):
        assert R.shape in [(2, 2), (3, 3)]
        dim = R.shape[0]
        eps = 1e-4
        assert np.all(np.isclose(R.T @ R, np.eye(dim), rtol=0.0, atol=eps))
        assert np.linalg.det(R) > 0
        if dim == 2:
            return np.arccos(R[0,0])
        elif dim == 3:
            return np.arccos(0.5 * (np.trace(R) - 1.0))
        else:
            assert False

    def _generate_sample(self):
        # Resample pose until properly inside image
        while True:
            R1, t1, R2, t2 = self._sample_pose_pair()
            # R1, t1 corresponds to reference image (observed)
            crop_box = self._bbox_from_projected_keypoints(R1, t1)
            width = crop_box[2] - crop_box[0]
            height = crop_box[3] - crop_box[1]
            if width >= 50 and height >= 50:
                # print("Success", crop_box)
                break
            # else:
            #     print("Fail", crop_box)
        # print("raw", crop_box)
        crop_box = self._wrap_bbox_in_squarebox(crop_box)
        # print("sq", crop_box)
        H = self._get_projectivity_for_crop_and_rescale(crop_box)
        K = H @ self._K

        img1 = pillow_to_pt(Image.fromarray(self._render(K, R1, t1)), normalize_flag=True, transform=self._aug_transform)
        img2 = pillow_to_pt(Image.fromarray(self._render(K, R2, t2)), normalize_flag=True, transform=self._aug_transform)
        # Dummy data:
        # img1 = torch.zeros([3] + list(self._configs.data.crop_dims))
        # img2 = torch.ones([3] + list(self._configs.data.crop_dims))

        data = img1, img2

        # How to rotate 2nd global frame, to align it with 1st global frame
        R21_global = R1 @ R2.T

        # Max values, after which ground truth is saturated
        MAX_DELTA_R33 = 2.0 # 2 when z flips 180 deg over
        MAX_PIXEL_OFFSET = 50.0
        MAX_DELTA_INPLANE = math.pi
        MAX_DELTA_THETA = math.pi * 0.5

        # NOTE ON RELATIVE DEPTH:
        # Actual depth is impossible to determine from image alone due to cropping effects on calibration.

        pixel_offset = pflat(K @ t2)[:2,0] - pflat(K @ t1)[:2,0]
        rel_depth_error = np.log(t2[2,0]) - np.log(t1[2,0])
        delta_angle_inplane = self._calc_delta_angle_inplane(R21_global)
        delta_theta = self._angle_from_rotmat(R21_global)
        delta_R33 = np.clip(-(1.0 + R21_global[2,2]) / MAX_DELTA_R33, 0.0, 1.0)
        norm_pixel_offset = np.clip(np.linalg.norm(pixel_offset) / MAX_PIXEL_OFFSET, 0.0, 1.0)
        abs_delta_angle_inplane = np.clip(np.abs(delta_angle_inplane) / MAX_DELTA_INPLANE, 0.0, 1.0)
        abs_delta_theta = np.clip(np.abs(delta_theta) / MAX_DELTA_THETA, 0.0, 1.0)

        annotation = Annotation(
            pixel_offset              = torch.tensor(pixel_offset).float(),
            rel_depth_error           = torch.tensor(rel_depth_error).float(),
            delta_angle_inplane       = torch.tensor(delta_angle_inplane).float(),
            delta_theta               = torch.tensor(delta_theta).float(),
            delta_R33                 = torch.tensor(delta_R33).float(),
            norm_pixel_offset         = torch.tensor(norm_pixel_offset).float(),
            abs_delta_angle_inplane   = torch.tensor(abs_delta_angle_inplane).float(),
            abs_delta_theta           = torch.tensor(abs_delta_theta).float(),
        )

        return data, annotation

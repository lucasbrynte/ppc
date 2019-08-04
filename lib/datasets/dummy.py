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
from lib.utils import uniform_sampling_on_S2, get_rotation_axis_angle, get_translation, sample_param
from lib.constants import TRAIN, VAL
from lib.loader import Sample
from lib.sixd_toolkit.pysixd import inout
from lib.rendering.glumpy_renderer import Renderer
from lib.rendering.pose_sampler import PoseSampler

ExtraInput = namedtuple('ExtraInput', [
    'crop_box_normalized',
])

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
        self.Targets = self._get_target_def()

        self._data_sampling_specs = getattr(self._configs.runtime.data_sampling, self._mode)
        assert len(self._data_sampling_specs) == 1, 'Mixing multiple dataset specs not supported as of yet'

        self._pids_path = '/tmp/sixd_kp_pids/' + self._mode
        if os.path.exists(self._pids_path):
            shutil.rmtree(self._pids_path)
            print("Removing " + self._pids_path)
        print("Creating " + self._pids_path)
        os.makedirs(self._pids_path)

    def _get_target_def(self):
        return namedtuple('Targets', list(self._configs.targets.keys()))

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

    def set_len(self, nbr_samples):
        self._len = nbr_samples

    def __len__(self):
        return self._len

    def _init_worker_seed(self):
        pid = os.getpid()
        pid_path = os.path.join(self._pids_path, str(pid))
        if os.path.exists(pid_path):
            return
        np.random.seed(pid)
        open(pid_path, 'w').close()

    def _at_epoch_start(self):
        self._init_worker_seed() # Cannot be called in constructor, since it is only executed by main process. Workaround: call at start of epoch.
        # self._renderer = self._init_renderer()

    def __getitem__(self, index):
        data, targets, extra_input = self._generate_sample()
        return Sample(targets, data, extra_input)

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

    def _expand_bbox(self, bbox, resize_factor):
        x1, y1, x2, y2 = bbox
        center_x = 0.5*(x1+x2)
        center_y = 0.5*(y1+y2)
        return (
            max(-0.5,                                  center_x + resize_factor*(x1 - center_x)),
            max(-0.5,                                  center_y + resize_factor*(y1 - center_y)),
            min(-0.5 + self._configs.data.img_dims[1], center_x + resize_factor*(x2 - center_x)),
            min(-0.5 + self._configs.data.img_dims[0], center_y + resize_factor*(y2 - center_y)),
        )

    def _bbox_from_projected_keypoints(self, R, t):
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        keypoints_3d = self._metadata['objects'][self._obj_label]['keypoints']
        keypoints_2d = pflat(self._K @ (R @ keypoints_3d + t))[:2,:]
        x1, y1 = np.min(keypoints_2d, axis=1)
        x2, y2 = np.max(keypoints_2d, axis=1)
        bbox = (x1, y1, x2, y2)
        bbox = self._expand_bbox(bbox, 1.5) # Expand bbox slightly, in order to include all of object (not just the keypoints)
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

    def _normalize_bbox(self, bbox, fx, fy, px, py):
        (x1, y1, x2, y2) = bbox
        x1 = (x1 - px) / fx
        y1 = (y1 - py) / fy
        x2 = (x2 - px) / fx
        y2 = (y2 - py) / fy
        bbox = (x1, y1, x2, y2)
        return bbox

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

    def _get_object_dimensions(self):
        obj_dimensions = np.diff(self._metadata['objects'][self._obj_label]['bbox3d'], axis=1).squeeze()
        assert obj_dimensions.shape == (3,)
        return obj_dimensions

    def _get_max_extent(self):
        """
        Returns maximum distance between 3D bbox corners, i.e. "diameter" of the box.
        """
        obj_dimensions = self._get_object_dimensions()
        max_extent = np.linalg.norm(0.5*obj_dimensions)
        return max_extent

    def _set_depth_by_translation_along_viewing_ray(self, T, new_depth):
        T = T.copy()
        old_depth = T[2,3]
        T[:3,3] *= new_depth / old_depth
        return T

    def _sample_perturbation_params(self):
        return {
            # 'axis_of_revolution': sample_param(self._data_sampling_specs[0].perturbation.axis_of_revolution),
            'axis_of_revolution': uniform_sampling_on_S2(),
            # Gaussian distribution
            'angle': np.pi/180. * sample_param(self._data_sampling_specs[0].perturbation.angle),
            'object_bias_over_extent': sample_param(self._data_sampling_specs[0].perturbation.object_bias_over_extent),
            # Log-normal distribution
            'depth_rescale_factor': sample_param(self._data_sampling_specs[0].perturbation.depth_rescale_factor),
        }

    def _sample_object_pose_params(self):
        table_size = self._data_sampling_specs[0].synthetic_ref.object_pose.table_size
        return {
            # 'object_azimuth_angle': sample_param(self._data_sampling_specs[0].object_pose.object_azimuth_angle),
            # 'xy_transl': sample_param(self._data_sampling_specs[0].object_pose.xy_transl),
            'object_azimuth_angle': np.random.uniform(low=0., high=2.*np.pi), # No reason to limit these perturbations - all angles allowed
            'xy_transl': np.random.uniform(low=-0.5*table_size, high=0.5*table_size, size=(2,)),
        }

    def _sample_camera_pose_params(self):
        return {
            'hemisphere_polar_angle': np.pi/180. * sample_param(self._data_sampling_specs[0].synthetic_ref.camera_pose.hemisphere_polar_angle),
            # 'hemisphere_azimuth_angle': np.pi/180. * sample_param(self._data_sampling_specs[0].synthetic_ref.camera_pose.hemisphere_azimuth_angle),
            'hemisphere_azimuth_angle': np.random.uniform(low=0., high=2.*np.pi), # No reason to limit these perturbations - all angles allowed
            'hemisphere_radius': sample_param(self._data_sampling_specs[0].synthetic_ref.camera_pose.hemisphere_radius),
            'inplane_rot_angle': np.pi/180. * sample_param(self._data_sampling_specs[0].synthetic_ref.camera_pose.inplane_rot_angle),
            'principal_axis_perturb_angle': np.pi/180. * sample_param(self._data_sampling_specs[0].synthetic_ref.camera_pose.principal_axis_perturb_angle),
            # 'inplane_angle_for_axis_of_revolution_for_paxis_perturb': np.pi/180. * sample_param(self._data_sampling_specs[0].synthetic_ref.camera_pose.inplane_angle_for_axis_of_revolution_for_paxis_perturb),
            'inplane_angle_for_axis_of_revolution_for_paxis_perturb': np.random.uniform(low=0., high=2.*np.pi), # No reason to limit these perturbations - all angles allowed
        }

    def _apply_perturbation(self, T1, perturb_params):
        # Map bias / extent ratio to actual translation:
        transl = self._get_object_dimensions() * perturb_params['object_bias_over_extent']

        # Note: perturbation in object frame. We want to apply rotations around object center rather than camera center (which would be quite uncontrolled).
        T_perturb_obj = get_translation(transl) @ get_rotation_axis_angle(perturb_params['axis_of_revolution'], perturb_params['angle'])
        T2 = T1 @ T_perturb_obj

        # Additional perturbation along viewing ray
        old_depth = T2[2,3]
        T2 = self._set_depth_by_translation_along_viewing_ray(T2, old_depth * perturb_params['depth_rescale_factor'])

        return T2

    def _sample_pose(self):
        pose_sampler = PoseSampler()
        up_dir = np.array([0., 0., 1.])
        zmin = self._metadata['objects'][self._obj_label]['bbox3d'][2,0]
        bottom_center = np.array([0., 0., zmin])

        # Sample parameters for an object pose such that the object is placed somewhere on the xy-plane.
        obj_pose_params = self._sample_object_pose_params()
        T_model2world = pose_sampler.calc_object_pose_on_xy_plane(obj_pose_params, up_dir, bottom_center)

        # Sample parameters for a camera pose.
        cam_pose_params = self._sample_camera_pose_params()
        T_world2cam = pose_sampler.calc_camera_pose(cam_pose_params)

        T1 = T_world2cam @ T_model2world
        return T1

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
        # Minimum allowed distance between object and camera centers
        min_dist_obj_and_camera_centers = self._get_max_extent() + self._data_sampling_specs[0].min_dist_obj_and_camera

        MAX_NBR_RESAMPLINGS = 100

        assert self._data_sampling_specs[0].ref_source == 'synthetic', 'Only synthetic ref images supported as of yet.'
        # Resample pose until accepted
        for j in range(MAX_NBR_RESAMPLINGS):
            # T1 corresponds to reference image (observed)
            T1 = self._sample_pose()
            R1 = T1[:3,:3]; t1 = T1[:3,[3]]

            if T1[2,3] < min_dist_obj_and_camera_centers:
                # print("Rejected T1, due to small depth", T1)
                continue

            crop_box = self._bbox_from_projected_keypoints(R1, t1)
            width = crop_box[2] - crop_box[0]
            height = crop_box[3] - crop_box[1]
            if width < 50 or height < 50:
                # print("Rejected T1, due to crop_box", crop_box)
                continue

            break
        else:
            assert False, '{}/{} resamplings performed, but no acceptable obj / cam pose was found'.format(MAX_NBR_RESAMPLINGS, MAX_NBR_RESAMPLINGS)

        # print("raw", crop_box)
        crop_box = self._wrap_bbox_in_squarebox(crop_box)
        # print("sq", crop_box)
        crop_box_normalized = self._normalize_bbox(crop_box, self._K[0,0], self._K[1,1], self._K[0,2], self._K[1,2])
        H = self._get_projectivity_for_crop_and_rescale(crop_box)
        K = H @ self._K

        # Resample perturbation until accepted
        for j in range(MAX_NBR_RESAMPLINGS):
            # Perturb reference T1, to get proposed pose T2
            perturb_params = self._sample_perturbation_params()
            T2 = self._apply_perturbation(T1, perturb_params)
            R2 = T2[:3,:3]; t2 = T2[:3,[3]]

            if T2[2,3] < min_dist_obj_and_camera_centers:
                # print("Rejected T2, due to small depth", T2)
                continue

            break
        else:
            assert False, '{}/{} resamplings performed, but no acceptable perturbation was found'.format(MAX_NBR_RESAMPLINGS, MAX_NBR_RESAMPLINGS)

        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T1[3,:], np.array([0., 0., 0., 1.])))
        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T2[3,:], np.array([0., 0., 0., 1.])))

        assert T1[2,3] > 0, "Center of object pose 1 behind camera"
        assert T2[2,3] > 0, "Center of object pose 2 behind camera"

        img1 = pillow_to_pt(Image.fromarray(self._render(K, R1, t1)), normalize_flag=True, transform=self._aug_transform)
        img2 = pillow_to_pt(Image.fromarray(self._render(K, R2, t2)), normalize_flag=True, transform=self._aug_transform)
        # Dummy data:
        # img1 = torch.zeros([3] + list(self._configs.data.crop_dims))
        # img2 = torch.ones([3] + list(self._configs.data.crop_dims))

        data = img1, img2

        # How to rotate 2nd global frame, to align it with 1st global frame
        R21_global = R1 @ R2.T

        # NOTE ON RELATIVE DEPTH:
        # Actual depth is impossible to determine from image alone due to cropping effects on calibration.

        pixel_offset = pflat(K @ t2)[:2,0] - pflat(K @ t1)[:2,0]
        delta_angle_inplane = self._calc_delta_angle_inplane(R21_global)
        delta_angle_total = self._angle_from_rotmat(R21_global)

        all_target_vals = {
            'pixel_offset': pixel_offset,
            'rel_depth_error': np.log(t2[2,0]) - np.log(t1[2,0]),
            'norm_pixel_offset': np.linalg.norm(pixel_offset),
            'delta_angle_inplane_signed': delta_angle_inplane,
            'delta_angle_inplane_unsigned': np.arccos(np.cos(delta_angle_inplane)), # cos & arccos combined will map angle to [0, pi] range
            'delta_angle_paxis': np.arccos(R21_global[2,2]),
            'delta_angle_total': delta_angle_total,
            'delta_angle_inplane_cosdist': 1.0 - np.cos(delta_angle_inplane),
            'delta_angle_paxis_cosdist': 1.0 - R21_global[2,2],
            'delta_angle_total_cosdist': 1.0 - np.cos(delta_angle_total),
        }

        target_vals = {key: val for key, val in all_target_vals.items() if key in self._configs.targets.keys()}

        for target_name, target_spec in self._configs.targets.items():
            target = target_vals[target_name]
            target = np.array(target)
            if len(target.shape) == 0:
                target = target[None] # Add redundant dimension (unsqueeze)
            if not (target_spec['min'] is None and target_spec['max'] is None):
                target = np.clip(target, target_spec['min'], target_spec['max'])
            target_vals[target_name] = torch.tensor(target).float()
        targets = self.Targets(**target_vals)

        extra_input = ExtraInput(
            crop_box_normalized = torch.tensor(crop_box_normalized).float(),
        )

        return data, targets, extra_input

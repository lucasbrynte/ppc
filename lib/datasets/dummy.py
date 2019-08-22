"""Reading input data in the SIXD common format."""
from collections import namedtuple
import os
import shutil
import glob
import ruamel.yaml as yaml
from attrdict import AttrDict

import random
import math
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset

from lib.utils import read_yaml_and_pickle, pextend, pflat, pillow_to_pt
from lib.utils import uniform_sampling_on_S2, get_rotation_axis_angle, get_translation, sample_param, calc_param_quantile_range, closest_rotmat
from lib.constants import TRAIN, VAL
from lib.loader import Sample
from lib.sixd_toolkit.pysixd import inout
from lib.rendering.glumpy_renderer import Renderer
from lib.rendering.pose_generation import calc_object_pose_on_xy_plane, calc_camera_pose

ExtraInput = namedtuple('ExtraInput', [
    'crop_box_normalized',
    'real_ref',
])

SampleMetaData = namedtuple('SampleMetaData', [
    'ref_img_path',
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
            'up_dir': np.array([0., 0., 1.]),
            'keypoints': rows2array(obj_anno, 'kp'),
            'kp_normals': rows2array(obj_anno, 'kp_normals'),
        } for obj_label, obj_anno in models_info.items()},
    }

def get_dataset(configs, mode, schemeset_name):
    return DummyDataset(configs, mode, schemeset_name)


global global_renderer, global_nyud_img_paths
global_renderer = None
global_nyud_img_paths = None

class DummyDataset(Dataset):
    def __init__(self, configs, mode, schemeset_name):
        self._configs = configs
        self._metadata = get_metadata(configs)
        self._mode = mode
        self._schemeset_name = schemeset_name
        self._K = self._read_calibration()
        self._models_info = self._init_models_info()
        self._obj_label = self._configs.obj_label
        self._obj_id = self._determine_obj_id(self._obj_label)
        self._models = self._init_models()
        self._renderer = self._init_renderer()
        self._nyud_img_paths = self._init_nyud_img_paths()
        self._aug_transform = None
        if self._mode == TRAIN and self._configs.data.ref_colorjitter_during_train:
            self._aug_transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03)
            # self._aug_transform = ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5), saturation=(0.7, 1.5), hue=(-0.03, 0.03))
        else:
            self._aug_transform = None
        self.Targets = self._get_target_def()

        self._data_sampling_scheme_defs = getattr(getattr(self._configs.runtime.data_sampling_scheme_defs, self._mode), schemeset_name)
        self._ref_sampling_schemes = getattr(getattr(self._configs.runtime.ref_sampling_schemes, self._mode), schemeset_name)
        self._query_sampling_scheme = getattr(getattr(self._configs.runtime.query_sampling_schemes, self._mode), schemeset_name)

        self._pids_path = '/tmp/sixd_kp_pids/{}_{}'.format(self._mode, schemeset_name)
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

    def _determine_obj_id(self, obj_label):
        filtered_obj_ids = [obj_id for obj_id, model_spec in self._models_info.items() if model_spec['readable_label'] == obj_label]
        assert len(filtered_obj_ids) == 1
        obj_id = filtered_obj_ids[0]
        return obj_id

    def _init_models(self):
        print("Loading models...")
        models = {}
        for obj_id in self._models_info:
            models[obj_id] = inout.load_ply(os.path.join(self._configs.data.path, 'models', 'obj_{:02}.ply'.format(obj_id)))
        print("Done.")
        return models

    def _init_renderer(self):
        global global_renderer
        if global_renderer is not None:
            print('Reusing renderer')
            return global_renderer
        renderer = Renderer(
            self._configs.data.crop_dims,
        )
        for obj_id, model in self._models.items():
            renderer._preprocess_object_model(obj_id, model)
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
        self._deterministic_perturbation_ranges = self._calc_deterministic_perturbation_ranges()

    def _calc_deterministic_perturbation_ranges(self):
        return {
            param_name: calc_param_quantile_range(AttrDict(sample_spec), len(self))
            for param_name, sample_spec in self._query_sampling_scheme.perturbation.items()
            if sample_spec['deterministic_quantile_range']
        }
    def __getitem__(self, sample_index_in_epoch):
        ref_scheme_idx = np.random.choice(len(self._data_sampling_scheme_defs.ref_schemeset), p=[scheme_def.sampling_prob for scheme_def in self._data_sampling_scheme_defs.ref_schemeset])
        data, targets, extra_input, meta_data = self._generate_sample(ref_scheme_idx, sample_index_in_epoch)
        return Sample(targets, data, extra_input, meta_data)

    def _render(self, K, R_list, t_list, instance_id_list, shading_params, T_world2cam=None, apply_bg=None, min_nbr_unoccluded_pixels=0, white_silhouette=False):
        if 'light_pos_worldframe' in shading_params:
            assert T_world2cam is not None
            light_pos_camframe = pflat(T_world2cam @ pextend(shading_params['light_pos_worldframe'].reshape((3,1)))).squeeze()[:3]
        else:
            # Default: camera origin
            light_pos_camframe = np.zeros((3,))
        assert light_pos_camframe.shape == (3,)
        rgb, depth, seg, instance_seg, normal_map, corr_map = self._renderer.render(
            K,
            R_list,
            t_list,
            instance_id_list,
            light_pos = light_pos_camframe,
            ambient_weight = shading_params['ambient_weight'],
            clip_near = 100, # mm
            clip_far = 10000, # mm
        )

        if np.sum(seg == self._obj_id) < min_nbr_unoccluded_pixels:
            return None

        if apply_bg is not None:
            assert not white_silhouette
            if np.random.random() < 0.5:
                # On BG & occluders:
                rgb[seg != self._obj_id] = apply_bg[seg != self._obj_id, :]
            else:
                # On BG only:
                rgb[seg == 0] = apply_bg[seg == 0, :]
        elif white_silhouette:
            rgb[seg == self._obj_id] = 255

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
        assert x1 >= 0 and x1 <= self._configs.data.img_dims[1]
        assert x2 >= 0 and x2 <= self._configs.data.img_dims[1]
        assert x1 >= 0 and y1 <= self._configs.data.img_dims[0]
        assert x2 >= 0 and y2 <= self._configs.data.img_dims[0]
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

    def _get_object_max_extent(self, obj_label, plane_normal=None):
        """
        Computes the distance between object center and the bounding box corner farthest away from the object center.
        If plane_normal is supplied, bbox corners will be projected to the corresponding plane before determining the maximum extent.
        """
        min_max_corners = self._metadata['objects'][obj_label]['bbox3d'].copy() # 2 columns, representing opposite corners
        if plane_normal is not None:
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            parallel_component = np.sum(min_max_corners * plane_normal[:,None], axis=0, keepdims=True) * plane_normal[:,None]
            min_max_corners -= parallel_component
        max_extent = np.max(np.linalg.norm(min_max_corners, axis=0))
        return max_extent

    def _sample_perturbation_params(self, sample_index_in_epoch):
        return {param_name: self._deterministic_perturbation_ranges[param_name][sample_index_in_epoch, ...] if sample_spec['deterministic_quantile_range'] else sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._query_sampling_scheme.perturbation.items()}

    def _sample_object_pose_params(self, ref_scheme_idx):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._ref_sampling_schemes[ref_scheme_idx].synth_opts.object_pose.items()}

    def _perturb_object_pose_params_for_occluder(self, base_params, obj_label_occluder, perturb_dir_angle_range=[0., 2*np.pi]):
        phi = np.random.uniform(low=perturb_dir_angle_range[0], high=perturb_dir_angle_range[1])
        xy_perturb_dir = np.array([np.cos(phi), np.sin(phi)])
        min_safe_cc_dist = self._get_object_max_extent(self._obj_label, plane_normal=self._metadata['objects'][self._obj_label]['up_dir']) + self._get_object_max_extent(obj_label_occluder, plane_normal=self._metadata['objects'][obj_label_occluder]['up_dir'])

        xy_perturb_magnitude = np.random.uniform(low=0.8*min_safe_cc_dist, high=1.1*min_safe_cc_dist)
        # xy_perturb_magnitude = min_safe_cc_dist

        xy_perturb = xy_perturb_dir * xy_perturb_magnitude
        return {
            'xy_transl': base_params['xy_transl'] + xy_perturb,
            'object_azimuth_angle': np.random.uniform(low=0, high=2*np.pi),
        }

    def _sample_camera_pose_params(self, ref_scheme_idx):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._ref_sampling_schemes[ref_scheme_idx].synth_opts.camera_pose.items()}

    def _sample_ref_shading_params(self, ref_scheme_idx):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._ref_sampling_schemes[ref_scheme_idx].synth_opts.shading.items()}

    def _sample_query_shading_params(self):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._query_sampling_scheme.shading.items()}

    def _apply_perturbation(self, T1, perturb_params):
        # Map bias / extent ratio to actual translation:
        transl = self._get_object_dimensions() * perturb_params['object_bias_over_extent']

        # Note: perturbation in object frame. We want to apply rotations around object center rather than camera center (which would be quite uncontrolled).
        T_perturb_obj = get_translation(transl) @ get_rotation_axis_angle(perturb_params['axis_of_revolution'], np.pi/180.*perturb_params['angle'])
        T2 = T1 @ T_perturb_obj

        # Additional perturbation along viewing ray
        old_depth = T2[2,3]
        T2 = self._set_depth_by_translation_along_viewing_ray(T2, old_depth * perturb_params['depth_rescale_factor'])

        return T2

    def _generate_object_pose(self, obj_pose_params, obj_label=None, occluder=False):
        if occluder:
            assert obj_label is not None
        elif obj_label is None:
            obj_label = self._obj_label
        obj_label = self._obj_label
        up_dir = self._metadata['objects'][obj_label]['up_dir']
        zmin = self._metadata['objects'][obj_label]['bbox3d'][2,0]
        bottom_center = np.array([0., 0., zmin])

        T_model2world = calc_object_pose_on_xy_plane(obj_pose_params, up_dir, bottom_center)

        return T_model2world

    def _sample_camera_pose(self, ref_scheme_idx):
        # Sample parameters for a camera pose.
        cam_pose_params = self._sample_camera_pose_params(ref_scheme_idx)
        T_world2cam = calc_camera_pose(cam_pose_params)

        return T_world2cam

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
        R = closest_rotmat(A)
        return self._angle_from_rotmat(R)

    def _clip_and_arccos(self, x):
        return np.arccos(np.clip(x, -1., 1.))

    def _angle_from_rotmat(self, R):
        assert R.shape in [(2, 2), (3, 3)]
        dim = R.shape[0]
        eps = 1e-3
        assert np.all(np.isclose(R.T @ R, np.eye(dim), rtol=0.0, atol=eps))
        assert np.linalg.det(R) > 0
        if dim == 2:
            return self._clip_and_arccos(R[0,0])
        elif dim == 3:
            return self._clip_and_arccos(0.5 * (np.trace(R) - 1.0))
        else:
            assert False

    def _check_seq_has_annotations_of_interest(self, root_path, sequence):
        global_info_path = os.path.join(root_path, sequence, 'global_info.yml')
        assert os.path.exists(global_info_path)
        global_info_yaml = self._read_yaml(global_info_path)
        return self._obj_label in global_info_yaml['obj_annotated_and_present']

    def _get_seq(self, ref_scheme_idx):
        seq = self._ref_sampling_schemes[ref_scheme_idx].real_opts.linemod_seq
        if '<OBJ_LABEL>' in seq:
            seq = seq.replace('<OBJ_LABEL>', self._obj_label)
        return seq

    def _crop_as_array(self, img, crop_box):
        (x1, y1, x2, y2) = crop_box
        img_mode = img.mode
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[y1:y2, x1:x2]
        else:
            assert len(img.shape) == 3
            img = img[y1:y2, x1:x2, :]
        img = Image.fromarray(img, mode=img_mode)
        return img

    def _read_img(self, ref_scheme_idx, crop_box, frame_idx, instance_idx, apply_bg=None):
        seq = self._get_seq(ref_scheme_idx)
        assert self._check_seq_has_annotations_of_interest(self._configs.data.path, seq), 'No annotations for sequence {}'.format(seq)

        rel_rgb_path = os.path.join(seq, 'rgb', str(frame_idx).zfill(6) + '.png')
        rgb_path = os.path.join(self._configs.data.path, rel_rgb_path)
        img = self._crop_as_array(Image.open(rgb_path), crop_box).resize(self._configs.data.crop_dims)
        if apply_bg is None \
                and not self._ref_sampling_schemes[ref_scheme_idx].white_silhouette \
                and not self._ref_sampling_schemes[ref_scheme_idx].real_opts.mask_silhouette:
            # Early return (no need to load seg)
            return img, rel_rgb_path
        seg_path = os.path.join(self._configs.data.path, seq, 'instance_seg', str(frame_idx).zfill(6) + '.png')
        seg = np.array(self._crop_as_array(Image.open(seg_path), crop_box).resize(self._configs.data.crop_dims))
        img_array = np.array(img)
        if apply_bg is not None:
            assert not self._ref_sampling_schemes[ref_scheme_idx].real_opts.mask_silhouette
            img_array[seg != instance_idx+1] = apply_bg[seg != instance_idx+1, :]
        elif self._ref_sampling_schemes[ref_scheme_idx].real_opts.mask_silhouette:
            img_array[seg != instance_idx+1] = 0
        if self._ref_sampling_schemes[ref_scheme_idx].white_silhouette:
            img_array[seg == instance_idx+1] = 255
        img = Image.fromarray(img_array, mode=img.mode)
        return img, rel_rgb_path

    def _read_pose_from_anno(self, ref_scheme_idx):
        seq = self._get_seq(ref_scheme_idx)
        assert self._check_seq_has_annotations_of_interest(self._configs.data.path, seq), 'No annotations for sequence {}'.format(seq)

        all_gts = self._read_yaml(os.path.join(self._configs.data.path, seq, 'gt.yml'))
        nbr_frames = len(all_gts)

        NBR_ATTEMPTS = 50
        for j in range(NBR_ATTEMPTS):
            if self._ref_sampling_schemes[ref_scheme_idx].real_opts.static_frame_idx is not None:
                frame_idx = self._ref_sampling_schemes[ref_scheme_idx].real_opts.static_frame_idx
            else:
                frame_idx = np.random.choice(nbr_frames)
            gts_in_frame = all_gts[frame_idx]

            # Filter annotated instances on object id
            enumerated_and_filtered_gts = [(instance_idx, gt) for instance_idx, gt in enumerate(gts_in_frame) if gt['obj_id'] == self._obj_id]
            nbr_instances = len(enumerated_and_filtered_gts)
            if nbr_instances == 0:
                continue
            instance_idx, gt = random.choice(enumerated_and_filtered_gts)

            R1 = closest_rotmat(np.array(gt['cam_R_m2c']).reshape((3, 3)))
            t1 = np.array(gt['cam_t_m2c']).reshape((3,1))

            crop_box = np.array(gt['obj_bb'])
            crop_box[2:] += crop_box[:2]  # x,y,w,h, -> x1,y1,x2,y2

            # crop_box = self._expand_bbox(crop_box, 1.5) # Expand crop_box slightly, in order to include all of object (not just the keypoints)
            # crop_box = self._truncate_bbox(crop_box)

            width = crop_box[2] - crop_box[0]
            height = crop_box[3] - crop_box[1]
            # print(width, height)
            if width < 25 or height < 25:
                # print("Rejected T1, due to crop_box", crop_box)
                continue

            break
        else:
            # print(gts_in_frame)
            # print(seq, frame_idx)
            assert False, 'After {} attempts, no frame found with annotations for desired object'.format(NBR_ATTEMPTS)

        return R1, t1, crop_box, frame_idx, instance_idx

    def _generate_synthetic_pose(self, ref_scheme_idx):
        # Resample pose until accepted
        for j in range(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings']):
            # T1 corresponds to reference image (observed)
            obj_pose_params = self._sample_object_pose_params(ref_scheme_idx)
            T_model2world = self._generate_object_pose(obj_pose_params)
            T_world2cam = self._sample_camera_pose(ref_scheme_idx)
            T1 = T_world2cam @ T_model2world
            R1 = T1[:3,:3]; t1 = T1[:3,[3]]

            # Minimum allowed distance between object and camera centers
            if T1[2,3] < self._get_max_extent() + self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['min_dist_obj_and_camera']:
                # print("Rejected T1, due to small depth", T1)
                continue

            crop_box = self._bbox_from_projected_keypoints(R1, t1)

            crop_box = self._expand_bbox(crop_box, 1.5) # Expand crop_box slightly, in order to include all of object (not just the keypoints)
            crop_box = self._truncate_bbox(crop_box)

            width = crop_box[2] - crop_box[0]
            height = crop_box[3] - crop_box[1]
            if width < 50 or height < 50:
                # print("Rejected T1, due to crop_box", crop_box)
                continue

            obj_labels_possible_occluders = [model_spec['readable_label'] for obj_id, model_spec in self._models_info.items() if obj_id != self._obj_id]
            T_occluders = {}
            nbr_occluders = 4
            for k, obj_label in enumerate(np.random.choice(obj_labels_possible_occluders, size=(nbr_occluders,), replace=False)):
                bin_size = 2*np.pi/nbr_occluders
                margin = 0.15 * bin_size
                perturb_dir_angle_range = k*bin_size + np.array([margin, bin_size - margin])
                T_occluders[obj_label] = T_world2cam @ self._generate_object_pose(self._perturb_object_pose_params_for_occluder(obj_pose_params, obj_label, perturb_dir_angle_range=perturb_dir_angle_range), obj_label=obj_label, occluder=True)

            break
        else:
            assert False, '{}/{} resamplings performed, but no acceptable obj / cam pose was found'.format(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'], self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'])

        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T1[3,:], np.array([0., 0., 0., 1.])))
        assert T1[2,3] > 0, "Center of object pose 1 behind camera"

        return T_model2world, T_occluders, T_world2cam, R1, t1, crop_box

    def _generate_perturbation(self, ref_scheme_idx, sample_index_in_epoch, R1, t1):
        T1 = np.eye(4)
        T1[:3,:3] = R1
        T1[:3,[3]] = t1

        # Resample perturbation until accepted
        for j in range(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings']):
            # Perturb reference T1, to get proposed pose T2
            perturb_params = self._sample_perturbation_params(sample_index_in_epoch)
            T2 = self._apply_perturbation(T1, perturb_params)
            R2 = T2[:3,:3]; t2 = T2[:3,[3]]

            # Minimum allowed distance between object and camera centers
            if T2[2,3] < self._get_max_extent() + self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['min_dist_obj_and_camera']:
                # print("Rejected T2, due to small depth", T2)
                continue

            break
        else:
            assert False, '{}/{} resamplings performed, but no acceptable perturbation was found'.format(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'], self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'])

        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T2[3,:], np.array([0., 0., 0., 1.])))
        assert T2[2,3] > 0, "Center of object pose 2 behind camera"

        return R2, t2

    def _sample_crop_box(self, img_height, img_width):
        x1 = np.random.randint(img_width - self._configs.data.crop_dims[1])
        x2 = x1 + self._configs.data.crop_dims[1]
        y1 = np.random.randint(img_height - self._configs.data.crop_dims[0])
        y2 = y1 + self._configs.data.crop_dims[0]
        return (x1, y1, x2, y2)

    def _init_nyud_img_paths(self):
        global global_nyud_img_paths
        if global_nyud_img_paths is not None:
            print('Reusing nyud_img_paths')
            return global_nyud_img_paths
        nyud_img_paths = glob.glob(os.path.join(self._configs.data.nyud_path, 'data', '**', 'r-*.ppm'))
        print('Not reusing nyud_img_paths')
        global_nyud_img_paths = nyud_img_paths
        return nyud_img_paths

    def _sample_nyud_patch(self):
        NBR_ATTEMPTS = 100
        for j in range(NBR_ATTEMPTS):
            path_idx = np.random.randint(len(self._nyud_img_paths))
            img_path = self._nyud_img_paths[path_idx]
            # img_path = os.path.join(self._configs.data.nyud_path, 'data/library_0005/r-1300707945.014378-1644637693.ppm')
            try:
                # NOTE: Some NYUD images are truncated, and for some reason this seems to cause an issue at Pillow crop
                # Unsure what will happen when cropping these iamges as numpy array, but try / catch kept as of now.
                full_img = Image.open(img_path)
                img_width, img_height = full_img.size
                return self._crop_as_array(full_img, self._sample_crop_box(img_height, img_width))
            except:
                # Not ideal to keep removing elements from long list...
                # Set would be tempting, but not straightforward to sample from
                del self._nyud_img_paths[path_idx]
                continue
        else:
            assert False, 'No proper NYU-D background image found'

    def _generate_sample(self, ref_scheme_idx, sample_index_in_epoch):
        # ======================================================================
        # NOTE: The reference pose / image is independent from the sample_index_in_epoch,
        # which only controls the perturbation, and thus the query image
        # ======================================================================

        # ref_scheme = self._data_sampling_scheme_defs.ref_schemeset[ref_scheme_idx].scheme_name
        # query_scheme = self._data_sampling_scheme_defs.query_scheme

        assert self._ref_sampling_schemes[ref_scheme_idx].ref_source in ['real', 'synthetic'], 'Unrecognized ref_source: {}.'.format(self._ref_sampling_schemes[ref_scheme_idx].ref_source)

        if self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real':
            R1, t1, crop_box, frame_idx, instance_idx = self._read_pose_from_anno(ref_scheme_idx)
        elif self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'synthetic':
            T1_model2world, T1_occluders, T_world2cam, R1, t1, crop_box = self._generate_synthetic_pose(ref_scheme_idx)

        # print("raw", crop_box)
        crop_box = self._wrap_bbox_in_squarebox(crop_box)
        # print("sq", crop_box)
        crop_box_normalized = self._normalize_bbox(crop_box, self._K[0,0], self._K[1,1], self._K[0,2], self._K[1,2])
        H = self._get_projectivity_for_crop_and_rescale(crop_box)
        K = H @ self._K

        R2, t2 = self._generate_perturbation(ref_scheme_idx, sample_index_in_epoch, R1, t1)

        if self._ref_sampling_schemes[ref_scheme_idx].sample_nyud_bg:
            ref_bg = np.array(self._sample_nyud_patch())
        else:
            ref_bg = None

        if self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real':
            img1, ref_img_path = self._read_img(ref_scheme_idx, crop_box, frame_idx, instance_idx, apply_bg=ref_bg)
        elif self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'synthetic':
            ref_shading_params = self._sample_ref_shading_params(ref_scheme_idx)
            R_list1, t_list1, instance_id_list1 = [R1], [t1], [self._obj_id]
            for obj_label, T in T1_occluders.items():
                R_list1.append(T[:3,:3])
                t_list1.append(T[:3,[3]])
                instance_id_list1.append(self._determine_obj_id(obj_label))
            img1 = self._render(K, R_list1, t_list1, instance_id_list1, ref_shading_params, T_world2cam=T_world2cam, apply_bg=ref_bg, min_nbr_unoccluded_pixels=200, white_silhouette=self._ref_sampling_schemes[ref_scheme_idx].white_silhouette)
            if img1 is None:
                print('Too few visible pixels - resampling via recursive call.')
                return self._generate_sample(ref_scheme_idx, sample_index_in_epoch)
            img1 = Image.fromarray(img1)
        query_shading_params = self._sample_query_shading_params()
        img2 = Image.fromarray(self._render(K, [R2], [t2], [self._obj_id], query_shading_params))

        # Augmentation + numpy -> pytorch conversion
        img1 = pillow_to_pt(img1, normalize_flag=True, transform=self._aug_transform)
        img2 = pillow_to_pt(img2, normalize_flag=True, transform=None)

        data = img1, img2

        # How to rotate 2nd global frame, to align it with 1st global frame
        R21_global = closest_rotmat(R1 @ R2.T)

        # NOTE ON RELATIVE DEPTH:
        # Actual depth is impossible to determine from image alone due to cropping effects on calibration.

        # Raise error instead of returning nan when calling arccos(x) for x outside of [-1, 1]
        with np.errstate(invalid='raise'):
            pixel_offset = pflat(K @ t2)[:2,0] - pflat(K @ t1)[:2,0]
            delta_angle_inplane = self._calc_delta_angle_inplane(R21_global)
            delta_angle_total = self._angle_from_rotmat(R21_global)

            all_target_vals = {
                'pixel_offset': pixel_offset,
                'rel_depth_error': np.log(t2[2,0]) - np.log(t1[2,0]),
                'norm_pixel_offset': np.linalg.norm(pixel_offset),
                'delta_angle_inplane_signed': delta_angle_inplane,
                'delta_angle_inplane_unsigned': self._clip_and_arccos(np.cos(delta_angle_inplane)), # cos & arccos combined will map angle to [0, pi] range
                'delta_angle_paxis': self._clip_and_arccos(R21_global[2,2]),
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
            real_ref = torch.tensor(self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real', dtype=torch.bool),
        )

        meta_data = SampleMetaData(
            ref_img_path = ref_img_path if self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real' else None,
        )

        return data, targets, extra_input, meta_data

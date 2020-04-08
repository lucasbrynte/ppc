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
import cv2 as cv
import torch
from torch import Tensor
from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset

from lib.utils import read_yaml_and_pickle, pextend, pflat, numpy_to_pt
# from lib.utils import get_eucl
from lib.utils import project_pts, uniform_sampling_on_S2, get_rotation_axis_angle, get_translation, sample_param, calc_param_quantile_range, closest_rotmat
from lib.utils import get_projectivity_for_crop_and_rescale_numpy, square_bbox_around_projected_object_center_numpy, crop_img
from lib.constants import TRAIN, VAL, TEST
from lib.loader import Sample
from lib.sixd_toolkit.pysixd import inout
from lib.rendering.glumpy_renderer import Renderer
from lib.rendering.pose_generation import calc_object_pose_on_xy_plane, calc_camera_pose

Maps = namedtuple('Maps', [
    'ref_img_full',
    'instance_seg1_full',
    'safe_fg_anno_mask_full',
])

ExtraInput = namedtuple('ExtraInput', [
    'real_ref',
    'K',
    'R1',
    't1',
    'R2',
    't2',
    'R2_init',
    't2_init',
    'obj_diameter',
    'ref_scheme_loss_weight',
    'query_scheme_loss_weight',
])

SampleMetaData = namedtuple('SampleMetaData', [
    'ref_img_path',
    'obj_id',
    # 'scheme_name',
    'tasks_punished',
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
            'diameter': obj_anno['diameter'],
            'up_dir': np.array([0., 0., 1.]),
            'keypoints': rows2array(obj_anno, 'kp'),
            'kp_normals': rows2array(obj_anno, 'kp_normals'),
        } for obj_label, obj_anno in models_info.items()},
    }

def get_dataset(configs, mode, schemeset_name):
    return DummyDataset(configs, mode, schemeset_name)


global global_renderer, global_nyud_img_paths, global_voc_img_paths
global_renderer = None
global_nyud_img_paths = None
global_voc_img_paths = None

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
        self._object_max_horizontal_extents = self._calc_object_max_horizontal_extents()
        self._renderer = self._init_renderer()
        self._nyud_img_paths = self._init_nyud_img_paths()
        self._voc_img_paths = self._init_voc_img_paths()
        if self._mode == TRAIN and not self._configs.data.ref_colorjitter_during_train is False:
            self._aug_transform = ColorJitter(brightness=0.3*0.4, contrast=0.3*0.4, saturation=0.3*0.4, hue=0.3*0.03)
            # self._aug_transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03)
            # self._aug_transform = ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5), saturation=(0.7, 1.5), hue=(-0.03, 0.03))
        self.Targets = self._get_target_def()

        self._data_sampling_scheme_defs = getattr(getattr(self._configs.runtime.data_sampling_scheme_defs, self._mode), schemeset_name)
        self._ref_sampling_schemes = getattr(getattr(self._configs.runtime.ref_sampling_schemes, self._mode), schemeset_name)
        self._query_sampling_schemes = getattr(getattr(self._configs.runtime.query_sampling_schemes, self._mode), schemeset_name)

        self.configure_sequences_and_lengths()

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
            self._configs.data.img_dims,
        )
        for obj_id, model in self._models.items():
            renderer._preprocess_object_model(obj_id, model)
        print('Not reusing renderer')
        global_renderer = renderer
        return renderer

    def configure_sequences_and_lengths(self):
        def check_obj_in_frame(gts_in_frame):
            enumerated_and_filtered_gts = [(instance_idx, gt) for instance_idx, gt in enumerate(gts_in_frame) if gt['obj_id'] == self._obj_id]
            nbr_instances = len(enumerated_and_filtered_gts)
            return nbr_instances > 0
        self._sequence_frames_filtered = []
        self._sequence_lengths = []
        for ref_scheme_idx in range(len(self._ref_sampling_schemes)):
            if self._ref_sampling_schemes[ref_scheme_idx].ref_source != 'real':
                self._sequence_frames_filtered.append(None)
                self._sequence_lengths.append(None)
                continue
            seq = self._get_seq(ref_scheme_idx)
            all_gts = self._read_yaml(os.path.join(self._configs.data.path, seq, 'gt.yml'))
            self._sequence_frames_filtered.append([ frame_idx for frame_idx in all_gts if check_obj_in_frame(all_gts[frame_idx]) ])
            self._sequence_lengths.append(len(self._sequence_frames_filtered[-1]))

    def set_deterministic_ref_scheme_sampling(self, flag):
        if flag:
            assert all(scheme.ref_source == 'real' for scheme in self._ref_sampling_schemes)
            self.set_len(sum(self._sequence_lengths))
        self._deterministic_ref_scheme_sampling = flag

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
        return [{
            param_name: calc_param_quantile_range(AttrDict(sample_spec), len(self))
            for param_name, sample_spec in query_scheme.perturbation.items()
            if sample_spec['deterministic_quantile_range']
        } for query_scheme in self._query_sampling_schemes]

    def _sample_idx_to_ref_scheme_idx_and_frame_idx(self, sample_idx):
        cumsum = np.cumsum(self._sequence_lengths)
        ref_scheme_idx = np.argwhere(cumsum > sample_idx)[0,0]
        frame_idx = sample_idx
        if ref_scheme_idx > 0:
            frame_idx -= cumsum[:ref_scheme_idx]
        return ref_scheme_idx, frame_idx

    # TODO: Add mode for deterministic looping over dataset. Set on schemeset level.
    # If set, set length to sum of samples over scheme sets.
    # Also implement mapping from sample index to schemeset / seq / frame
    # Finally allow nbr_batches = None
    def __getitem__(self, sample_index_in_epoch):
        if self._deterministic_ref_scheme_sampling:
            ref_scheme_idx, fixed_frame_idx = self._sample_idx_to_ref_scheme_idx_and_frame_idx(sample_index_in_epoch)
        else:
            ref_scheme_idx = np.random.choice(len(self._data_sampling_scheme_defs.ref_schemeset), p=[scheme_def.sampling_prob for scheme_def in self._data_sampling_scheme_defs.ref_schemeset])
            fixed_frame_idx = None
        if self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['loading']['coupled_ref_and_query_scheme_sampling']:
            # Ref & query schemes are sampled jointly. Lists need to be of same length to be able to map elements.
            assert len(query_sampling_scheme_list) == len(ref_sampling_scheme_list)
            query_scheme_idx = ref_scheme_idx
        else:
            query_scheme_idx = np.random.choice(len(self._data_sampling_scheme_defs.query_schemeset), p=[scheme_def.sampling_prob for scheme_def in self._data_sampling_scheme_defs.query_schemeset])
        R1, t1, R2_init, t2_init, ref_img_path, img1, instance_seg1, safe_fg_anno_mask = self._generate_ref_img_and_anno(ref_scheme_idx, query_scheme_idx, sample_index_in_epoch, fixed_frame_idx=fixed_frame_idx)
        # Augmentation + numpy -> pytorch conversion
        if self._configs.data.ref_colorjitter_during_train is True or (self._configs.data.ref_colorjitter_during_train == 'synthonly' and self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'synthetic'):
            img1 = np.array(self._aug_transform(Image.fromarray(img1, mode='RGB')))


        # TODO: When in eval_poseeopt mode, all of the following can be omitted / replaced by dummy operations - since they are done online anyway:
        # - Generating perturbation.
        # - Rendering query img. (Furthermore - during inference real data could possibly be assumed, in which case glumpy renderer may somehow be reserved for the online rendering. This is faster than neural renderer.)
        # - Defining crop box & corresponding transformation H
        # - Cropping & resizing ref image
        # - Target calculation
        # - 
        R2, t2 = self._generate_perturbation(ref_scheme_idx, query_scheme_idx, sample_index_in_epoch, R1, t1)

        # Compute crop box and everything in order to determine H, which depends on the query pose.
        xc, yc, width, height = square_bbox_around_projected_object_center_numpy(t2, self._K, self._metadata['objects'][self._obj_label]['diameter'], crop_box_resize_factor = self._configs.data.crop_box_resize_factor)
        H = get_projectivity_for_crop_and_rescale_numpy(xc, yc, width, height, self._configs.data.crop_dims)

        sample = self._generate_sample(
            ref_scheme_idx,
            query_scheme_idx,
            sample_index_in_epoch,
            H,
            R1,
            t1,
            ref_img_path,
            img1,
            instance_seg1,
            safe_fg_anno_mask,
            R2,
            t2,
            R2_init,
            t2_init,
        )

        if self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['pushopt']:
            # In the neural renderer case, rendering has not yet been carried out. The placeholders im2 and instance_seg2 remain unchanged.
            assert self._configs.data.query_rendering_method == 'neural'
            sample_at_opt = self._generate_sample(
                ref_scheme_idx,
                query_scheme_idx,
                sample_index_in_epoch,
                H,
                R1,
                t1,
                ref_img_path,
                img1,
                instance_seg1,
                safe_fg_anno_mask,
                R1, # Ref pose sent in as query pose!
                t1, # Ref pose sent in as query pose!
                R2_init,
                t2_init,
            )
            pushopt_prob = self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['pushopt_prob']
            if pushopt_prob is None:
                return [sample, sample_at_opt]
            else:
                if np.random.random() <= pushopt_prob:
                    return [sample_at_opt]
                else:
                    return [sample]
        else:
            return [sample]

    def _get_ref_bg(self, ref_scheme_idx, bg_dims, black_already=False):
        if self._ref_sampling_schemes[ref_scheme_idx].background == 'nyud':
            return self._sample_bg_patch(self._nyud_img_paths, bg_dims)
        elif self._ref_sampling_schemes[ref_scheme_idx].background == 'voc':
            return self._sample_bg_patch(self._voc_img_paths, bg_dims)
        elif self._ref_sampling_schemes[ref_scheme_idx].background == 'black':
            return np.zeros(list(bg_dims)+[3], dtype=np.uint8) if not black_already else None
        assert self._ref_sampling_schemes[ref_scheme_idx].background is None
        return None

    def _apply_bg(self, rgb, instance_seg, bg, inplace=False):
        if not inplace:
            rgb = rgb.copy()
        # if False:
        if np.random.random() < 0.5:
            # On BG & occluders:
            rgb[instance_seg != 1] = bg[instance_seg != 1, :]
        else:
            # On BG only:
            rgb[instance_seg == 0] = bg[instance_seg == 0, :]
        return rgb

    def _set_white_silhouette(self, rgb, instance_seg, inplace=False):
        if not inplace:
            rgb = rgb.copy()
        rgb[instance_seg == 1] = 255
        return rgb

    def _render(self, K, R, t, obj_id, R_occluders_list, t_occluders_list, obj_id_occluders, shading_params, trunc_dims=None, T_world2cam=None, min_nbr_unoccluded_pixels=0):
        if 'light_pos_worldframe' in shading_params:
            assert T_world2cam is not None
            light_pos_camframe = pflat(T_world2cam @ pextend(shading_params['light_pos_worldframe'].reshape((3,1)))).squeeze()[:3]
        else:
            # Default: camera origin
            light_pos_camframe = np.zeros((3,))
        assert light_pos_camframe.shape == (3,)
        rgb, depth, seg, instance_seg, normal_map, corr_map = self._renderer.render(
            K,
            [R] + R_occluders_list,
            [t] + t_occluders_list,
            [obj_id] + obj_id_occluders,
            light_pos = light_pos_camframe,
            ambient_weight = shading_params['ambient_weight'],
            diffuse_weight = shading_params['diffuse_weight'] if 'diffuse_weight' in shading_params else (1.0 - shading_params['ambient_weight']),
            specular_weight = shading_params['specular_weight'] if 'specular_weight' in shading_params else 0.0,
            specular_shininess = shading_params['specular_shininess'] if 'specular_shininess' in shading_params else 3.0,
            specular_whiteness = shading_params['specular_whiteness'] if 'specular_whiteness' in shading_params else 0.3,
            clip_near = 100, # mm
            clip_far = 10000, # mm
        )

        if trunc_dims is not None:
            height, width = trunc_dims
            rgb = rgb[:height, :width, :]
            depth = depth[:height, :width]
            seg = seg[:height, :width]
            instance_seg = instance_seg[:height, :width]
            normal_map = normal_map[:height, :width, :]
            corr_map = corr_map[:height, :width, :]

        # instance_seg is 0 on BG, 1 on object of interest, and 2 on occluders

        if np.sum(instance_seg == 1) < min_nbr_unoccluded_pixels:
            return None, None

        return rgb, instance_seg

    # def _truncate_bbox(self, bbox):
    #     (x1, y1, x2, y2) = bbox
    #     x1, x2 = np.clip((x1, x2), 0, self._configs.data.img_dims[1])
    #     y1, y2 = np.clip((y1, y2), 0, self._configs.data.img_dims[0])
    #     truncated_bbox = (x1, y1, x2, y2)
    #     return truncated_bbox

    def _sample_bbox_shift(self, bbox, max_rel_shift_factor=0.3):
        (x1, y1, x2, y2) = bbox
        width = x2 - x1
        height = y2 - y1
        assert width <= self._configs.data.img_dims[1]
        assert height <= self._configs.data.img_dims[0]
        rel_xshift = np.random.uniform(-max_rel_shift_factor, max_rel_shift_factor)
        rel_yshift = np.random.uniform(-max_rel_shift_factor, max_rel_shift_factor)
        return rel_xshift*width, rel_yshift*height

    def _shift_bbox(self, bbox, xshift, yshift):
        (x1, y1, x2, y2) = bbox
        assert x2 - x1 <= self._configs.data.img_dims[1]
        assert y2 - y1 <= self._configs.data.img_dims[0]
        x1, x2 = x1+xshift, x2+xshift
        y1, y2 = y1+yshift, y2+yshift
        shifted_bbox = (x1, y1, x2, y2)
        return shifted_bbox

    # def _shift_bbox_into_img(self, bbox):
    #     (x1, y1, x2, y2) = bbox
    #     assert x2 - x1 <= self._configs.data.img_dims[1]
    #     assert y2 - y1 <= self._configs.data.img_dims[0]
    # 
    #     if x1 < 0:
    #         xshift = 0 - x1
    #     elif x2 > self._configs.data.img_dims[1]:
    #         xshift = - (x2 - self._configs.data.img_dims[1])
    #     else:
    #         xshift = 0
    # 
    #     if y1 < 0:
    #         yshift = 0 - y1
    #     elif y2 > self._configs.data.img_dims[0]:
    #         yshift = - (y2 - self._configs.data.img_dims[0])
    #     else:
    #         yshift = 0
    # 
    #     shifted_bbox = self._shift_bbox(bbox, xshift, yshift)
    # 
    #     (x1, y1, x2, y2) = shifted_bbox
    #     assert x1 >= 0 and x1 <= self._configs.data.img_dims[1]
    #     assert x2 >= 0 and x2 <= self._configs.data.img_dims[1]
    #     assert x1 >= 0 and y1 <= self._configs.data.img_dims[0]
    #     assert x2 >= 0 and y2 <= self._configs.data.img_dims[0]
    # 
    #     return shifted_bbox

    # def _resize_bbox(self, bbox, resize_factor):
    #     x1, y1, x2, y2 = bbox
    #     center_x = 0.5*(x1+x2)
    #     center_y = 0.5*(y1+y2)
    #     return (
    #         max(-0.5,                                  center_x + resize_factor*(x1 - center_x)),
    #         max(-0.5,                                  center_y + resize_factor*(y1 - center_y)),
    #         min(-0.5 + self._configs.data.img_dims[1], center_x + resize_factor*(x2 - center_x)),
    #         min(-0.5 + self._configs.data.img_dims[0], center_y + resize_factor*(y2 - center_y)),
    #     )

    # def _wrap_bbox_in_squarebox(self, bbox):
    #     (x1_old, y1_old, x2_old, y2_old) = bbox
    #     old_width = x2_old - x1_old
    #     old_height = y2_old - y1_old
    #     if old_width < old_height:
    #         new_height = old_height
    #         new_width  = old_height
    #     else:
    #         new_width  = old_width
    #         new_height = old_width
    #     delta_height = new_height - old_height
    #     delta_width = new_width - old_width
    #     assert delta_height >= 0
    #     assert delta_width >= 0
    #     xc = (x1_old + x2_old) * 0.5
    #     yc = (y1_old + y2_old) * 0.5
    #     square_bbox = gen_bbox(xc, yc, new_width, new_height)
    #     return square_bbox

    def _get_object_dimensions(self):
        obj_dimensions = np.diff(self._metadata['objects'][self._obj_label]['bbox3d'], axis=1).squeeze()
        assert obj_dimensions.shape == (3,)
        return obj_dimensions

    def _get_obj_diameter(self):
        return self._metadata['objects'][self._obj_label]['diameter']

    def _set_depth_by_translation_along_viewing_ray(self, T, new_depth):
        T = T.copy()
        old_depth = T[2,3]
        T[:3,3] *= new_depth / old_depth
        return T

    def _calc_object_max_horizontal_extent(self, obj_label):
        """
        Computes the distance between object center and the bounding box corner farthest away from the object center.
        If plane_normal is supplied, bbox corners will be projected to the corresponding plane before determining the maximum extent.
        """
        plane_normal = self._metadata['objects'][obj_label]['up_dir'].copy()
        pts = self._models[self._determine_obj_id(obj_label)]['pts'].copy().T
        if plane_normal is not None:
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            parallel_component = np.sum(pts * plane_normal[:,None], axis=0, keepdims=True) * plane_normal[:,None]
            pts -= parallel_component
        max_extent = np.max(np.linalg.norm(pts, axis=0))
        return max_extent

    def _calc_object_max_horizontal_extents(self):
        return { obj_label: self._calc_object_max_horizontal_extent(obj_label) for obj_label in self._metadata['objects'].keys() }

    # def _get_object_max_extent(self, obj_label, plane_normal=None):
    #     """
    #     Computes the distance between object center and the bounding box corner farthest away from the object center.
    #     If plane_normal is supplied, bbox corners will be projected to the corresponding plane before determining the maximum extent.
    #     """
    #     min_max_corners = self._metadata['objects'][obj_label]['bbox3d'].copy() # 2 columns, representing opposite corners
    #     if plane_normal is not None:
    #         plane_normal = plane_normal / np.linalg.norm(plane_normal)
    #         parallel_component = np.sum(min_max_corners * plane_normal[:,None], axis=0, keepdims=True) * plane_normal[:,None]
    #         min_max_corners -= parallel_component
    #     max_extent = np.max(np.linalg.norm(min_max_corners, axis=0))
    #     return max_extent

    def _sample_perturbation_params(self, query_scheme_idx, sample_index_in_epoch):
        return {param_name: self._deterministic_perturbation_ranges[query_scheme_idx][param_name][sample_index_in_epoch, ...] if sample_spec['deterministic_quantile_range'] else sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._query_sampling_schemes[query_scheme_idx].perturbation.items()}

    def _sample_object_pose_params(self, ref_scheme_idx):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._ref_sampling_schemes[ref_scheme_idx].synth_opts.object_pose.items()}

    def _perturb_object_pose_params_for_occluder(self, base_params, obj_label_occluder, perturb_dir_angle_range=[0., 2*np.pi]):
        phi = np.random.uniform(low=perturb_dir_angle_range[0], high=perturb_dir_angle_range[1])
        xy_perturb_dir = np.array([np.cos(phi), np.sin(phi)])

        # min_safe_cc_dist = self._get_object_max_extent(self._obj_label, plane_normal=self._metadata['objects'][self._obj_label]['up_dir']) + self._get_object_max_extent(obj_label_occluder, plane_normal=self._metadata['objects'][obj_label_occluder]['up_dir'])
        min_safe_cc_dist = self._object_max_horizontal_extents[self._obj_label] \
                         + self._object_max_horizontal_extents[obj_label_occluder]
        # xy_perturb_magnitude = np.random.uniform(low=0.8*min_safe_cc_dist, high=1.1*min_safe_cc_dist)
        xy_perturb_magnitude = min_safe_cc_dist

        xy_perturb = xy_perturb_dir * xy_perturb_magnitude
        return {
            'xy_transl': base_params['xy_transl'] + xy_perturb,
            'object_azimuth_angle': np.random.uniform(low=0, high=2*np.pi),
        }

    def _sample_camera_pose_params(self, ref_scheme_idx):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._ref_sampling_schemes[ref_scheme_idx].synth_opts.camera_pose.items()}

    def _sample_ref_shading_params(self, ref_scheme_idx):
        return {param_name: sample_param(AttrDict(sample_spec)) for param_name, sample_spec in self._ref_sampling_schemes[ref_scheme_idx].synth_opts.shading.items()}

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

    def _generate_object_pose(self, obj_pose_params, obj_label=None, occluder=False, z_bias_range=None):
        if occluder:
            assert obj_label is not None
        elif obj_label is None:
            obj_label = self._obj_label
        up_dir = self._metadata['objects'][obj_label]['up_dir']
        zmin = self._metadata['objects'][obj_label]['bbox3d'][2,0]
        bottom_center = np.array([0., 0., zmin])

        T_model2world = calc_object_pose_on_xy_plane(obj_pose_params, up_dir, bottom_center, z_bias_range=z_bias_range)

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
        assert self._check_seq_has_annotations_of_interest(self._configs.data.path, seq), 'No annotations for sequence {}'.format(seq)
        return seq

    def _read_img(self, seq, frame_idx):
        rel_rgb_path = os.path.join(seq, 'rgb', str(frame_idx).zfill(6) + '.png')
        rgb_path = os.path.join(self._configs.data.path, rel_rgb_path)
        img = np.array(Image.open(rgb_path))
        assert tuple(img.shape[:2]) == self._configs.data.img_dims

        return img, rel_rgb_path

    def _read_instance_seg(self, seq, frame_idx, instance_idx):
        # Load instance segmentation
        instance_seg_path = os.path.join(self._configs.data.path, seq, 'instance_seg', str(frame_idx).zfill(6) + '.png')
        instance_seg_raw = np.array(Image.open(instance_seg_path))
        assert tuple(instance_seg_raw.shape[:2]) == self._configs.data.img_dims

        # Map indices to 0 / 1 / 2
        instance_seg = np.empty_like(instance_seg_raw)
        instance_seg.fill(2) # Default: occluder
        instance_seg[instance_seg_raw == 0] = 0 # Preserve index 0 for BG
        instance_seg[instance_seg_raw == instance_idx+1] = 1 # instance_idx+1 -> 1 (obj_of_interest)

        return instance_seg

    def _get_depth_map(self, seq, frame_idx, depth_scale=1e-3, rendered=False):
        """
        Reads observed / rendered depth map stored as a 16-bit image, and converts to an float32 array in meters.
        depth_scale: multiply the depth map with this factor to get depth in m
        """
        subdir = 'depth_rendered' if rendered else 'depth'
        depth_path = os.path.join(self._configs.data.path, seq, subdir, str(frame_idx).zfill(6) + '.png')

        # NOTE! simplify this back again, once the root of the PNG issue is found...
        # depth_map = np.array(Image.open(depth_path), dtype=np.uint16)
        try:
            # depth_path = '/datasets/occluded-linemod-augmented/all_unoccl/duck/depth_rendered/000487.png'
            # depth_path = '/datasets/occluded-linemod-augmented/all_unoccl/duck/depth_rendered/000872.png'
            depth_map11 = Image.open(depth_path)
            # depth_map = np.array(depth_map11, dtype=np.uint16)
            # NOTE: weirdly, calling np.array() with the dtype=np.uint16 argument may seemingly randomly generate errors with PngImageFile -> int conversion, but the same thing does not happen with initial np.array() call followed by .astype(np.uint16).
            depth_map11b = np.array(depth_map11)
            depth_map = depth_map11b.astype(np.uint16)
        except:
            print(depth_path)
            print(depth_map11)

            no_dtype_enforced = np.array(depth_map11)
            print(no_dtype_enforced.shape)
            print(no_dtype_enforced.dtype)
            print(no_dtype_enforced)

            asarray = np.asarray(depth_map11)
            print(asarray.shape)
            print(asarray)

            tmp1 = no_dtype_enforced.astype(np.uint16)
            tmp2 = asarray.astype(np.uint16)

            try:
                print(depth_map)
            except:
                depth_map = depth_map11b.astype(np.uint16)
                print(depth_map)
            assert False
        assert tuple(depth_map.shape[:2]) == self._configs.data.img_dims

        depth_map = depth_map.astype(np.float32) * float(depth_scale)
        return depth_map

    # def _compute_gt_optflow(self, depth_map_rendered, silhouette_mask, K, R1, t1, R2, t2):
    #     # Normalize pixels
    #     # Backproject normalized 2D points
    #     # Reproject into 2nd camera & compute residual
    # 
    #     # NOTE: Use fundamental matrix instead..?
    #     # NOTE: Use fundamental matrix instead..?
    #     # NOTE: Use fundamental matrix instead..?
    #     # NOTE: Use fundamental matrix instead..?
    #     # NOTE: Use fundamental matrix instead..?
    # 
    #     return gt_optflow

    def _get_safe_fg_anno_mask(self, depth_map, depth_map_rendered):
        """
        Reads observed depth (from RGB-D sensor) and rendered depth, and returns a mask such that:
        (1) The observed depth is positive (i.e. not corrupted / missing data)
        (2) The depths match up to a certain threshold
        The idea is that in the case of an unannotated occluder, the depths would not match.
        """
        MIN_DEPTH_FOR_VALID_OBS = 0.05 # 5 cm
        DEPTH_TH = 0.02 # At most 2 cm depth discrepancy for pixel to be safely annotated (no unannotated occlusion detected)
        safe_fg_anno_mask = (depth_map >= MIN_DEPTH_FOR_VALID_OBS) & (np.abs(depth_map - depth_map_rendered) < DEPTH_TH)

        return safe_fg_anno_mask

    def _read_pose_from_anno(self, ref_scheme_idx, fixed_frame_idx=None):
        seq = self._get_seq(ref_scheme_idx)

        all_gts = self._read_yaml(os.path.join(self._configs.data.path, seq, 'gt.yml'))
        # NOTE: Last frame missing (from both data and annotations). Lost during pre-processing scripts in ~/research/3dod/preprocessing/rigidpose, due to wrong "sequence_length" entries in ~/object-pose-estimation/meta.json
        # nbr_frames = len(all_gts)

        all_infos = self._read_yaml(os.path.join(self._configs.data.path, seq, 'info.yml'))

        NBR_ATTEMPTS = 50
        for j in range(NBR_ATTEMPTS):
            if self._ref_sampling_schemes[ref_scheme_idx].real_opts.static_frame_idx is not None:
                frame_idx = self._ref_sampling_schemes[ref_scheme_idx].real_opts.static_frame_idx
            elif fixed_frame_idx is not None:
                frame_idx = fixed_frame_idx
                frame_idx = self._sequence_frames_filtered[ref_scheme_idx][frame_idx]
            else:
                frame_idx = np.random.choice(self._sequence_lengths[ref_scheme_idx])
                frame_idx = self._sequence_frames_filtered[ref_scheme_idx][frame_idx]
            gts_in_frame = all_gts[frame_idx]

            # Verify that calibration matrix is indeed constant, and the same as the one annotated in camera.yml
            curr_K_anno = np.array(all_infos[frame_idx]['cam_K']).reshape((3, 3))
            assert np.all(np.isclose(curr_K_anno, self._K))

            # Filter annotated instances on object id
            enumerated_and_filtered_gts = [(instance_idx, gt) for instance_idx, gt in enumerate(gts_in_frame) if gt['obj_id'] == self._obj_id]
            nbr_instances = len(enumerated_and_filtered_gts)
            if nbr_instances == 0:
                if self._ref_sampling_schemes[ref_scheme_idx].real_opts.static_frame_idx is not None or fixed_frame_idx is not None:
                    assert False, 'No {} instance found in seq {}, frame {}'.format(self._obj_label, seq, frame_idx)
                continue
            instance_idx, gt = random.choice(enumerated_and_filtered_gts)

            R1 = closest_rotmat(np.array(gt['cam_R_m2c']).reshape((3, 3)))
            t1 = np.array(gt['cam_t_m2c']).reshape((3,1))

            if self._mode == TEST and 'init_pose_suffix' in self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['poseopt']:
                R2_init = closest_rotmat(np.array(gt['cam_R_m2c{}'.format(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['poseopt']['init_pose_suffix'])]).reshape((3, 3)))
                t2_init = np.array(gt['cam_t_m2c{}'.format(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['poseopt']['init_pose_suffix'])]).reshape((3,1))
            else:
                R2_init = R1
                t2_init = t1

            break
        else:
            # print(gts_in_frame)
            # print(seq, frame_idx)
            assert False, 'After {} attempts, no frame found with annotations for desired object'.format(NBR_ATTEMPTS)

        return R1, t1, R2_init, t2_init, frame_idx, instance_idx

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
            if T1[2,3] < 0.5*self._get_obj_diameter() + self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['min_dist_obj_and_camera']:
                # print("Rejected T1, due to small depth", T1)
                continue

            # Verify object center projected within image.
            assert t1.shape == (3,1)
            assert self._K.shape == (3,3)
            obj_cc_proj = pflat(self._K @ t1)[:2,:]
            if not np.all((obj_cc_proj >= 0) & (obj_cc_proj <= np.array(self._configs.data.img_dims)[[1,0],None])): # Flip indices for img_dims to be in xy rather than yx order.
                # print("Rejected T1, due to object center projected outside image", obj_cc_proj)
                continue

            obj_labels_possible_occluders = [model_spec['readable_label'] for obj_id, model_spec in self._models_info.items() if obj_id != self._obj_id]
            T_occluders = {}
            nbr_occluders = self._ref_sampling_schemes[ref_scheme_idx].synth_opts.occluders.nbr_occluders
            if nbr_occluders > 0:
                if self._ref_sampling_schemes[ref_scheme_idx].synth_opts.occluders.method == 'surrounding_main_obj':
                    for k, obj_label in enumerate(np.random.choice(obj_labels_possible_occluders, size=(nbr_occluders,), replace=False)):
                        bin_size = 2*np.pi/nbr_occluders
                        margin = 0.15 * bin_size
                        perturb_dir_angle_range = k*bin_size + np.array([margin, bin_size - margin])
                        T_occluders[obj_label] = T_world2cam @ self._generate_object_pose(self._perturb_object_pose_params_for_occluder(obj_pose_params, obj_label, perturb_dir_angle_range=perturb_dir_angle_range), obj_label=obj_label, occluder=True, z_bias_range=self._ref_sampling_schemes[ref_scheme_idx].synth_opts.occluders.z_bias_range)
                elif self._ref_sampling_schemes[ref_scheme_idx].synth_opts.occluders.method == 'in_front_of_main_obj':
                    for k, obj_label in enumerate(np.random.choice(obj_labels_possible_occluders, size=(nbr_occluders,), replace=False)):
                        t_perturb = np.zeros((3,))
                        main_diameter = self._metadata['objects'][self._obj_label]['diameter']
                        this_diameter = self._metadata['objects'][obj_label]['diameter']
                        avg_diameter = 0.5*(main_diameter + this_diameter)
                        # z_diff = 0.5*(main_diameter + this_diameter) # Touch but don't collide
                        z_diff = np.random.uniform(
                            low = 0.5*(main_diameter + this_diameter), # At least don't collide
                            high = 0.3, # Max 30 cm
                        )
                        main_depth = T1[2,3]
                        this_depth = main_depth - z_diff
                        main_diameter_proj = main_diameter / main_depth
                        this_diameter_proj = this_diameter / this_depth
                        # xy_diff = np.random.uniform(
                        #     low = 0.5*(this_diameter_proj-main_diameter_proj) + 0.3*main_diameter_proj, # First align silhouette edges, and then uncover at least 30% of main obj diameter.
                        #     high = 0.5*(main_diameter_proj + this_diameter_proj), # Until no more occl
                        # )
                        xy_diff = np.random.uniform(
                            # low = 0.0,
                            low = 0.5*(this_diameter_proj-main_diameter_proj) if this_diameter > main_diameter else 0.0, # First align silhouette edges, and then uncover at least 30% of main obj diameter.
                            high = 0.5*(main_diameter_proj + this_diameter_proj), # Until no more occl
                        )
                        theta = np.random.uniform(low=0., high=2*np.pi)
                        t_perturb[:2] = xy_diff * this_depth * np.array([np.cos(theta), np.sin(theta)])
                        t_perturb -= z_diff * T1[:3,3] / np.linalg.norm(T1[:3,3]) # Move along viewing direction
                        T_occluders[obj_label] = get_translation(t_perturb) @ T1 @ get_rotation_axis_angle(uniform_sampling_on_S2(), np.random.uniform(low=0.0, high=np.pi))
                else:
                    assert False

            break
        else:
            assert False, '{}/{} resamplings performed, but no acceptable obj / cam pose was found'.format(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'], self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'])

        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T1[3,:], np.array([0., 0., 0., 1.])))
        assert T1[2,3] > 0, "Center of object pose 1 behind camera"

        return T_model2world, T_occluders, T_world2cam, R1, t1

    def _generate_perturbation(self, ref_scheme_idx, query_scheme_idx, sample_index_in_epoch, R1, t1):
        T1 = np.eye(4)
        T1[:3,:3] = R1
        T1[:3,[3]] = t1

        # Resample perturbation until accepted
        for j in range(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings']):
            # Perturb reference T1, to get proposed pose T2
            perturb_params = self._sample_perturbation_params(query_scheme_idx, sample_index_in_epoch)
            T2 = self._apply_perturbation(T1, perturb_params)
            R2 = T2[:3,:3]; t2 = T2[:3,[3]]

            # Minimum allowed distance between object and camera centers
            if T2[2,3] < 0.5*self._get_obj_diameter() + self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['min_dist_obj_and_camera']:
                # print("Rejected T2, due to small depth", T2)
                continue

            break
        else:
            assert False, '{}/{} resamplings performed, but no acceptable perturbation was found'.format(self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'], self._configs.runtime.data_sampling_scheme_defs[self._mode][self._schemeset_name]['opts']['data']['max_nbr_resamplings'])

        # Last rows expected to remain unchanged:
        assert np.all(np.isclose(T2[3,:], np.array([0., 0., 0., 1.])))
        assert T2[2,3] > 0, "Center of object pose 2 behind camera"

        return R2, t2

    def _sample_crop_box(self, full_img_dims, crop_dims):
        assert full_img_dims[0] >= crop_dims[0]
        assert full_img_dims[1] >= crop_dims[1]
        x1 = 0 if full_img_dims[1] == crop_dims[1] else np.random.randint(full_img_dims[1] - crop_dims[1])
        x2 = x1 + crop_dims[1]
        y1 = 0 if full_img_dims[0] == crop_dims[0] else np.random.randint(full_img_dims[0] - crop_dims[0])
        y2 = y1 + crop_dims[0]
        return (x1, y1, x2, y2)

    def _init_nyud_img_paths(self):
        global global_nyud_img_paths
        if global_nyud_img_paths is not None:
            print('Reusing nyud_img_paths')
            return global_nyud_img_paths
        nyud_img_paths = glob.glob(os.path.join(self._configs.data.nyud_path, 'data', '**', 'r-*.ppm'))
        assert len(nyud_img_paths) > 0
        print('Not reusing nyud_img_paths')
        global_nyud_img_paths = nyud_img_paths
        return nyud_img_paths

    def _init_voc_img_paths(self):
        global global_voc_img_paths
        if global_voc_img_paths is not None:
            print('Reusing voc_img_paths')
            return global_voc_img_paths
        voc_img_paths = glob.glob(os.path.join(self._configs.data.voc_path, 'JPEGImages', '*.jpg'))
        assert len(voc_img_paths) > 0
        print('Not reusing voc_img_paths')
        global_voc_img_paths = voc_img_paths
        return voc_img_paths

    def _sample_bg_patch(self, bg_img_paths, bg_crop_dims):
        NBR_ATTEMPTS = 100
        for j in range(NBR_ATTEMPTS):
            path_idx = np.random.randint(len(bg_img_paths))
            img_path = bg_img_paths[path_idx]
            # img_path = os.path.join(self._configs.data.nyud_path, 'data/library_0005/r-1300707945.014378-1644637693.ppm')
            img = Image.open(img_path)
            img = np.array(img)
            yreps = -( -self._configs.data.img_dims[0] // img.shape[0] ) # Ceiling integer division
            xreps = -( -self._configs.data.img_dims[1] // img.shape[1] ) # Ceiling integer division
            full_img = np.tile(img, (yreps, xreps, 1))
            return crop_img(full_img, self._sample_crop_box(full_img.shape[:2], bg_crop_dims), pad_if_outside=False)
        else:
            assert False, 'No proper background image found'

    def _blur_obj(self, img, instance_seg, blur_opts, inplace=False, sigma_rescale_factor=1.0):
        if not inplace:
            img = img.copy()

        sigma = np.random.uniform(low=sigma_rescale_factor*blur_opts.sigma_range[0], high=sigma_rescale_factor*blur_opts.sigma_range[1])
        # Compute where to truncate kernel using inverse of OpenCV getGaussianKernel() default behavior
        ksize = 1 + 2 * ( 1 + max(0, math.ceil((sigma-0.8)/0.3)) )
        assert ksize >= 3
        assert ksize % 2 == 1
        img_blurred = cv.GaussianBlur(img, (ksize,ksize), sigma)

        silhouette = (instance_seg == 1).astype(np.uint8)
        kernel = np.ones((3,3), dtype=np.uint8)

        if blur_opts.mode == 'on_border_only':
            # Compute morphological gradient
            blur_region_uint8 = cv.morphologyEx(silhouette, cv.MORPH_GRADIENT, kernel)
        elif blur_opts.mode == 'dilated_silhouette':
            # Just dilate the silhouette
            blur_region_uint8 = cv.dilate(silhouette, kernel, iterations=blur_opts.dilate_nbr_px_margin)
        else:
            assert blur_opts.mode == 'silhouette'
            # Just apply on silhouette as-is
            blur_region_uint8 = silhouette

        blur_region = blur_region_uint8.astype(np.bool)
        assert np.allclose(blur_region, blur_region)

        img[blur_region] = img_blurred[blur_region]

        return img

    def _dims_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return (y2-y1, x2-x1)

    def _get_ref_img_real(self, ref_scheme_idx, frame_idx, instance_idx):
        ref_bg = self._get_ref_bg(ref_scheme_idx, self._configs.data.img_dims, black_already=False)
        seq = self._get_seq(ref_scheme_idx)
        img1, ref_img_path = self._read_img(seq, frame_idx)
        instance_seg1 = self._read_instance_seg(seq, frame_idx, instance_idx)

        # Determine depth scale and read depth maps
        all_infos = self._read_yaml(os.path.join(self._configs.data.path, seq, 'info.yml'))
        depth_scale = 1e-3 * all_infos[frame_idx]['depth_scale'] # Multiply with 1e-3 to convert to meters instead of mm.
        depth_map = self._get_depth_map(seq, frame_idx, depth_scale=depth_scale, rendered=False)
        depth_map_rendered = self._get_depth_map(seq, frame_idx, depth_scale=depth_scale, rendered=True)

        # Determine at what foreground pixels the annotated segmentation can be relied upon
        safe_fg_anno_mask = self._get_safe_fg_anno_mask(depth_map, depth_map_rendered)
        safe_fg_anno_mask &= instance_seg1 == instance_idx+1

        return img1, instance_seg1, ref_bg, safe_fg_anno_mask, ref_img_path

    def _get_ref_img_synthetic(self, ref_scheme_idx, R1, t1, T1_occluders, T_world2cam):
        ref_bg = self._get_ref_bg(ref_scheme_idx, self._configs.data.img_dims, black_already=True)
        ref_shading_params = self._sample_ref_shading_params(ref_scheme_idx)
        R_occluders_list1, t_occluders_list1, obj_id_occluders_list1 = [], [], []
        for obj_label, T in T1_occluders.items():
            R_occluders_list1.append(T[:3,:3])
            t_occluders_list1.append(T[:3,[3]])
            obj_id_occluders_list1.append(self._determine_obj_id(obj_label))
        img1, instance_seg1 = self._render(self._K, R1, t1, self._obj_id, R_occluders_list1, t_occluders_list1, obj_id_occluders_list1, ref_shading_params, trunc_dims=self._configs.data.img_dims, T_world2cam=T_world2cam, min_nbr_unoccluded_pixels=200)#self._configs.data.synth_ref_min_nbr_unoccluded_pixels)
        if img1 is None:
            return None

        # Determine at what foreground pixels the annotated segmentation can be relied upon
        safe_fg_anno_mask = instance_seg1 == 1

        return img1, instance_seg1, ref_bg, safe_fg_anno_mask

    def _calc_targets(self, H, R1, t1, R2, t2):
        HK = H @ self._K

        # How to rotate 2nd camera frame, to align it with 1st camera frame
        R21_cam = closest_rotmat(R1 @ R2.T)

        # NOTE ON RELATIVE DEPTH:
        # Actual depth is impossible to determine from image alone due to cropping effects on calibration.

        # Raise error instead of returning nan when calling arccos(x) for x outside of [-1, 1]
        with np.errstate(invalid='raise'):
            pixel_offset = pflat(HK @ t2)[:2,0] - pflat(HK @ t1)[:2,0]
            delta_angle_inplane = self._calc_delta_angle_inplane(R21_cam)
            delta_angle_total = self._angle_from_rotmat(R21_cam)

            total_nbr_model_pts = self._models[self._obj_id]['pts'].shape[0]
            # sampled_nbr_model_pts = 3000
            # sampled_model_pts = self._models[self._obj_id]['pts'][np.random.choice(total_nbr_model_pts, size=sampled_nbr_model_pts), :]
            sampled_model_pts = self._models[self._obj_id]['pts']
            avg_reproj_err = np.mean(np.linalg.norm(project_pts(sampled_model_pts.T, HK, R2, t2) - project_pts(sampled_model_pts.T, HK, R1, t1), axis=0))
            all_target_vals = {
                'avg_reproj_err': avg_reproj_err,
                'pixel_offset': pixel_offset,
                'rel_depth_error': np.log(t2[2,0]) - np.log(t1[2,0]),
                'norm_pixel_offset': np.linalg.norm(pixel_offset),
                'delta_angle_inplane_signed': delta_angle_inplane,
                'delta_angle_inplane_unsigned': self._clip_and_arccos(np.cos(delta_angle_inplane)), # cos & arccos combined will map angle to [0, pi] range
                'delta_angle_paxis': self._clip_and_arccos(R21_cam[2,2]),
                'delta_angle_total': delta_angle_total,
                'delta_angle_inplane_cosdist': 1.0 - np.cos(delta_angle_inplane),
                'delta_angle_paxis_cosdist': 1.0 - R21_cam[2,2],
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
        return self.Targets(**target_vals)

    def _generate_ref_img_and_anno(self, ref_scheme_idx, query_scheme_idx, sample_index_in_epoch, fixed_frame_idx=None):
        # ======================================================================
        # NOTE: The reference pose / image is independent from the sample_index_in_epoch,
        # which only controls the perturbation, and thus the query image
        # ======================================================================

        # ref_scheme = self._data_sampling_scheme_defs.ref_schemeset[ref_scheme_idx].scheme_name
        # query_scheme = self._data_sampling_scheme_defs.query_schemeset[query_scheme_idx].scheme_name

        assert self._ref_sampling_schemes[ref_scheme_idx].ref_source in ['real', 'synthetic'], 'Unrecognized ref_source: {}.'.format(self._ref_sampling_schemes[ref_scheme_idx].ref_source)

        if self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real':
            R1, t1, R2_init, t2_init, frame_idx, instance_idx = self._read_pose_from_anno(ref_scheme_idx, fixed_frame_idx=fixed_frame_idx)
        elif self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'synthetic':
            T1_model2world, T1_occluders, T_world2cam, R1, t1 = self._generate_synthetic_pose(ref_scheme_idx)
            # In synthetic case, set initial pose (the proposal) to ground truth
            R2_init, t2_init = R1, t1

        if self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real':
            img1, instance_seg1, ref_bg, safe_fg_anno_mask, ref_img_path = self._get_ref_img_real(ref_scheme_idx, frame_idx, instance_idx)
        elif self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'synthetic':
            ret = self._get_ref_img_synthetic(ref_scheme_idx, R1, t1, T1_occluders, T_world2cam)
            if ret is None:
                print('Too few visible pixels - resampling via recursive call.')
                return self._generate_ref_img_and_anno(ref_scheme_idx, query_scheme_idx, sample_index_in_epoch, fixed_frame_idx=fixed_frame_idx)
            else:
                img1, instance_seg1, ref_bg, safe_fg_anno_mask = ret
                ref_img_path = None
        else:
            assert False

        # Ref BG & silhouette post-processing
        if ref_bg is not None:
            img1 = self._apply_bg(img1, instance_seg1, ref_bg)
        if self._ref_sampling_schemes[ref_scheme_idx].white_silhouette:
            img1 = self._set_white_silhouette(img1, instance_seg1)

        # Apply blurring
        for blur_opts in self._ref_sampling_schemes[ref_scheme_idx].blur_chain:
            # NOTE: At least blurring the edges is probably quite important for synthetic images, since they are rendered without any anti-aliasing mechanism.
            if blur_opts.sigma_rescale_based_on_silhouette_extent:
                # More or less ensures that potential silhouette dilation after upsampling is invariant to depth and size of object.
                sigma_rescale_factor = 10. * self._metadata['objects'][self._obj_label]['diameter'] / t1[2,0]
            else:
                sigma_rescale_factor = 1.
            img1 = self._blur_obj(
                img1,
                instance_seg1,
                blur_opts,
                sigma_rescale_factor = sigma_rescale_factor,
            )

        return R1, t1, R2_init, t2_init, ref_img_path, img1, instance_seg1, safe_fg_anno_mask

    def _generate_sample(
            self,
            ref_scheme_idx,
            query_scheme_idx,
            sample_index_in_epoch,
            H,
            R1,
            t1,
            ref_img_path,
            img1,
            instance_seg1,
            safe_fg_anno_mask,
            R2,
            t2,
            R2_init,
            t2_init,
        ):
        # # Transformation from 1st camera frame to 2nd camera frame
        # T12_cam = get_eucl(R2, t2) @ get_eucl(R1.T, -R1.T@t1)
        # # T12_cam = get_eucl(R2, t2) @ np.linalg.inv(get_eucl(R1, t1))

        # # Compute ground truth optical flow
        # gt_optflow = self._compute_gt_optflow(depth_map_rendered, instance_seg1 == 1, HK, R1, t1, R2, t2)

        # Convert maps to pytorch tensors, and organize in attrdict
        maps = Maps(
            ref_img_full = numpy_to_pt(img1.astype(np.float32), normalize_flag=True),
            instance_seg1_full = numpy_to_pt(instance_seg1.astype(np.int64), normalize_flag=False),
            safe_fg_anno_mask_full = numpy_to_pt(safe_fg_anno_mask.astype(np.bool), normalize_flag=False),
        )

        targets = self._calc_targets(H, R1, t1, R2, t2)

        extra_input = ExtraInput(
            real_ref = torch.tensor(self._ref_sampling_schemes[ref_scheme_idx].ref_source == 'real', dtype=torch.bool),
            K = torch.tensor(self._K, dtype=torch.float32),
            R1 = torch.tensor(R1, dtype=torch.float32),
            t1 = torch.tensor(t1, dtype=torch.float32),
            R2 = torch.tensor(R2, dtype=torch.float32),
            t2 = torch.tensor(t2, dtype=torch.float32),
            R2_init = torch.tensor(R2_init, dtype=torch.float32),
            t2_init = torch.tensor(t2_init, dtype=torch.float32),
            obj_diameter = torch.tensor(self._metadata['objects'][self._obj_label]['diameter'], dtype=torch.float32),
            ref_scheme_loss_weight = torch.tensor(self._data_sampling_scheme_defs.ref_schemeset[ref_scheme_idx].loss_weight, dtype=torch.float32),
            query_scheme_loss_weight = torch.tensor(self._data_sampling_scheme_defs.query_schemeset[query_scheme_idx].loss_weight, dtype=torch.float32),
        )

        meta_data = SampleMetaData(
            ref_img_path = ref_img_path,
            obj_id = self._obj_id,
            # scheme_name = self._data_sampling_scheme_defs.query_schemeset[query_scheme_idx].scheme_name,
            tasks_punished = self._data_sampling_scheme_defs.query_schemeset[query_scheme_idx].tasks_punished,
        )

        return Sample(targets, maps, extra_input, meta_data)

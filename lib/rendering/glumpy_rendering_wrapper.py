import os
import ruamel.yaml as yaml
import numpy as np
import torch
from torch import nn
from lib.constants import TV_MEAN, TV_STD
from lib.utils import read_yaml_and_pickle
from lib.sixd_toolkit.pysixd import inout
from lib.rendering.glumpy_renderer import Renderer

from PIL import Image

class GlumpyRenderingWrapper():
    def __init__(self, configs, max_render_dims=(480,640)):
        self._configs = configs
        self._max_render_dims = max_render_dims
        self._init_render_opts()
        self._models_info = self._init_models_info()
        self._obj_label = self._configs.obj_label
        self._obj_id = self._determine_obj_id(self._obj_label)
        self._models = self._init_models()
        self._renderer = self._init_renderer()

    def _init_render_opts(self):
        self._clip_near = 10 # mm
        self._clip_far = 10000 # mm

    def _init_models_info(self):
        return read_yaml_and_pickle(os.path.join(self._configs.data.path, 'models', 'models_info.yml'))

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

    def get_model_info(self, obj_id):
        return self._models_info[obj_id]

    def get_model_pts(self, obj_id, numpy_mode=False):
        model_pts = self._models[obj_id]['pts']
        if not numpy_mode:
            model_pts = torch.tensor(model_pts).float().cuda()
        return model_pts

    def _init_renderer(self):
        renderer = Renderer(
            self._max_render_dims,
        )
        for obj_id, model in self._models.items():
            renderer._preprocess_object_model(obj_id, model)
        return renderer

    def render(
        self,
        HK,
        R,
        t,
        obj_id,
        ambient_weight,
        render_dims,
        R_occluders_list = [],
        t_occluders_list = [],
        obj_id_occluders = [],
        light_pos_camframe = [0,0,0],
        diffuse_weight = 0.0,
        specular_weight = 0.0,
        specular_shininess = 3.0,
        specular_whiteness = 0.3,
        lowres_render_dims=None,
        numpy_mode = False,
        batched = True,
    ):
        if lowres_render_dims is not None:
            assert render_dims[0] % lowres_render_dims[0] == 0
            lowres_render_factor = render_dims[0] // lowres_render_dims[0]
            assert render_dims[1] % lowres_render_dims[1] == 0
            assert lowres_render_factor == render_dims[1] // lowres_render_dims[1]
        else:
            lowres_render_factor = 1
            lowres_render_dims = render_dims
        assert lowres_render_dims[0] <= self._max_render_dims[0] and lowres_render_dims[1] <= self._max_render_dims[1]

        nbr_occluders = len(R_occluders_list)
        assert len(t_occluders_list) == nbr_occluders
        assert len(obj_id_occluders) == nbr_occluders

        if nbr_occluders > 0:
             # Not implemented
            assert numpy_mode
            assert not batched

        light_pos_camframe = np.array(light_pos_camframe)
        assert light_pos_camframe.shape == (3,)

        if not numpy_mode:
            device = HK.device
            HK = HK.detach().cpu().numpy()
            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()

        if not batched:
            HK = HK[None,:,:]
            R = R[None,:,:]
            t = t[None,:,:]
            obj_id = [obj_id]
            ambient_weight = [ambient_weight]

        batch_size = HK.shape[0]
        assert R.shape[0] == batch_size
        assert t.shape[0] == batch_size
        assert len(obj_id) == batch_size
        assert len(ambient_weight) == batch_size

        if lowres_render_factor != 1:
            HK[:,:2,:] /= lowres_render_factor

        height, width = lowres_render_dims
        img = torch.empty((batch_size, 3, height, width))
        instance_seg = torch.empty((batch_size, 1, height, width))
        for sample_idx in range(batch_size):
            buffers = self._renderer.render(
                HK[sample_idx,:,:],
                [R[sample_idx,:,:]] + R_occluders_list,
                [t[sample_idx,:,:]] + t_occluders_list,
                [obj_id[sample_idx]] + obj_id_occluders,
                render_dims = render_dims,
                light_pos = light_pos_camframe,
                ambient_weight = ambient_weight[sample_idx],
                diffuse_weight = diffuse_weight,
                specular_weight = specular_weight,
                specular_shininess = specular_shininess,
                specular_whiteness = specular_whiteness,
                clip_near = self._clip_near, # mm
                clip_far = self._clip_far, # mm
                desired_buffers = ['rgb', 'instance_seg'],
            )

            curr_rgb = buffers['rgb'][:height, :width, :]
            # curr_depth = buffers['depth'][:height, :width]
            # curr_seg = buffers['seg'][:height, :width]
            curr_instance_seg = buffers['instance_seg'][:height, :width]
            # curr_normal_map = buffers['normal_map'][:height, :width, :]
            # curr_corr_map = buffers['corr_map'][:height, :width, :]

            img[sample_idx,:,:,:] = torch.tensor(curr_rgb.astype(np.float32)).permute(2,0,1)
            instance_seg[sample_idx,0,:,:] = torch.tensor(curr_instance_seg.astype(np.int64))

        if not numpy_mode:
            img = img.to(device)
            instance_seg = instance_seg.to(device)

        if lowres_render_dims != render_dims:
            # Not entirely sure whether align_corners should be True or False. It should relate to renderer discretization & image coordinate frame convention. Might be that OpenGL has origin at center of top-left pixel, while neural renderer has it at top-left corner of top-left pixel, but not entirely sure.
            # From experiments with heavy upsampling, it seems like if align_corners=True, discretization should also have an effect on the factor by which HK is rescaled, otherwise object will be too enlarged.
            img = nn.functional.upsample(img, self._configs.data.crop_dims, mode='bilinear', align_corners=False)
            instance_seg = nn.functional.upsample(instance_seg.float(), self._configs.data.crop_dims, mode='nearest').long()

        if not batched:
            img = img.squeeze(0)
            instance_seg = instance_seg.squeeze(0)

        if numpy_mode:
            assert not batched # Not implemented
            img = img.permute(1,2,0).numpy().astype(np.uint8)
            instance_seg = instance_seg.permute(1,2,0).numpy().astype(np.uint8)

        return {
            'img': img,
            'instance_seg': instance_seg,
        }

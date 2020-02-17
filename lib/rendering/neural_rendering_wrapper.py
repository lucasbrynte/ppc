import os
import ruamel.yaml as yaml
import numpy as np
import torch
from lib.constants import TV_MEAN, TV_STD
from lib.utils import read_yaml_and_pickle
from lib.sixd_toolkit.pysixd import inout

# import sys
# sys.path.insert(0, '/workspace/lib/neural_renderer/build/lib.linux-x86_64-3.7')
import neural_renderer as nr
# from lib.neural_renderer import neural_renderer as nr


class NeuralRenderingWrapper():
    def __init__(self, configs):
        self._configs = configs
        self._init_render_opts()
        self._models_info = self._init_models_info()
        self._obj_label = self._configs.obj_label
        self._obj_id = self._determine_obj_id(self._obj_label)
        self._models = self._init_models()
        self._renderer = self._init_renderer()

    def _init_render_opts(self):
        self._texture_size = 2
        assert self._configs.data.crop_dims[0] == self._configs.data.crop_dims[1]
        self._render_size = self._configs.data.crop_dims[0]
        self._light_dir_camframe = [0,0,1]

    def _init_models_info(self):
        return read_yaml_and_pickle(os.path.join(self._configs.data.path, 'models', 'models_info.yml'))

    def _determine_obj_id(self, obj_label):
        filtered_obj_ids = [obj_id for obj_id, model_spec in self._models_info.items() if model_spec['readable_label'] == obj_label]
        assert len(filtered_obj_ids) == 1
        obj_id = filtered_obj_ids[0]
        return obj_id

    def _create_textures(self, faces, vtx_colors):
        face_colors = vtx_colors[faces,:] # shape: (num_faces, 3 (vertices 0-2), RGB)

        textures = np.zeros((1, faces.shape[0], self._texture_size, self._texture_size, self._texture_size, 3), dtype=np.float32)
        for i1, w1 in enumerate(np.linspace(0.0, 1.0, self._texture_size)):
            for i2, w2 in enumerate(np.linspace(0.0, 1.0, self._texture_size)):
                for i3, w3 in enumerate(np.linspace(0.0, 1.0, self._texture_size)):
                    if np.all(np.array([i1, i2, i3]) == 0):
                        w = np.ones((3,))
                        #textures[:, :, i1, i2, i3, :] = np.ones((1,1,1))
                        #continue
                    else:
                        w = np.array([w1, w2, w3])
                    w /= w.sum()
                    textures[:, :, i1, i2, i3, :] = np.sum(w[None, None, :, None] * face_colors[None, :, :, :], axis=2)
        return textures

    def _init_models(self):
        print("Loading models for neural renderer...")
        models = {}
        for obj_id in self._models_info:
            if obj_id != self._obj_id:
                # Only this object will be used (at least for this time being)
                continue

            model_np_dict = inout.load_ply(os.path.join(self._configs.data.path, 'models', 'obj_{:02}.ply'.format(obj_id)))

            vertices = model_np_dict['pts'].astype(np.float32)
            vtx_colors = model_np_dict['colors'].astype(np.float32) / 255.
            faces = model_np_dict['faces'].astype(np.int32)

            # Create per-face 3D texture maps in barycentric coordinates
            textures = self._create_textures(faces, vtx_colors)

            # numpy -> pytorch & GPU
            vertices = torch.tensor(vertices[None, :,:]).cuda()
            textures = torch.tensor(textures).cuda()
            faces = torch.tensor(faces[None, :,:]).cuda()

            models[obj_id] = {
                'vertices': vertices,
                'faces': faces,
                'textures': textures,
            }
        print("Done.")
        return models

    def _init_renderer(self):
        return nr.Renderer(
            image_size=self._render_size,
            anti_aliasing=True,
            background_color=[0,0,0],
            fill_back=True,
            camera_mode='projection',
            # K=None, # set at rendering
            # R=None, # set at rendering
            # t=None, # set at rendering
            dist_coeffs=None, # Default. Radial distorsion?
            orig_size=self._render_size,
            # perspective=True,
            # viewing_angle=30,
            # camera_direction=[0,0,1],
            near=100, # mm
            far=10000, # mm
            # light_intensity_ambient=0.5, # set at rendering
            # light_intensity_directional=0.5, # set at rendering
            light_color_ambient=[1,1,1],
            light_color_directional=[1,1,1],
            # light_direction=[0,1,0], # set at rendering
        )

    def render(self, HK, R, t, obj_id, ambient_weight):
        # Configure ambient / diffuse components weighting
        assert len(set(ambient_weight)) == 1
        ambient_weight = ambient_weight[0]
        self._renderer.light_intensity_ambient = ambient_weight
        self._renderer.light_intensity_directional = 1.0 - ambient_weight

        # Map lighting direction from camera to object frame
        light_dir_objframe = torch.matmul(R.permute(0, 2, 1), torch.tensor(self._light_dir_camframe, dtype=torch.float32).reshape((1, 3, 1)).cuda()).squeeze(2)
        self._renderer.light_direction = light_dir_objframe

        # Find desired model
        bs = len(obj_id)
        if len(set(obj_id)) == 1:
            # Repeatedly copying this single model may be avoided by torch.expand
            curr_obj_id = obj_id[0]
            vertices = self._models[curr_obj_id]['vertices'].expand(bs,-1,-1)
            faces = self._models[curr_obj_id]['faces'].expand(bs,-1,-1)
            textures = self._models[curr_obj_id]['textures'].expand(bs,-1,-1,-1,-1,-1)
        else:
            vertices = torch.cat([ self._models[curr_obj_id]['vertices'] for curr_obj_id in obj_id ], dim=0)
            faces = torch.cat([ self._models[curr_obj_id]['faces'] for curr_obj_id in obj_id ], dim=0)
            textures = torch.cat([ self._models[curr_obj_id]['textures'] for curr_obj_id in obj_id ], dim=0)

        # images, _, _ = self._renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        img = self._renderer(vertices, faces, textures, mode='rgb', K=HK, R=R, t=t.permute((0,2,1)))  # [batch_size, RGB, image_size, image_size]
        img = (255.*img - TV_MEAN[None,:,None,None].cuda()) / TV_STD[None,:,None,None].cuda()

        return img

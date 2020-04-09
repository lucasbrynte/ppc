# Python / glumpy OpenGL renderer.
# Inspiration & pieces of code taken from https://github.com/thodan/sixd_toolkit/blob/master/pysixd/renderer.py

import numpy as np
from glumpy import app, gloo, gl

# Set backend (http://glumpy.readthedocs.io/en/latest/api/app-backends.html)
app.use('glfw')
# app.use('qt5')
# app.use('pyside')

# Set logging level
from glumpy.log import log
import logging
log.setLevel(logging.WARNING) # ERROR, WARNING, DEBUG, INFO

class Renderer():
    _vertex_shader = """
    uniform mat4 u_mv;
    uniform mat4 u_nm;
    uniform mat4 u_mvp;
    uniform vec3 u_light_eye_pos;

    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    in vec2 in_texcoord;

    out vec3 vs_color;
    out vec2 vs_texcoord;
    out vec3 vs_obj_pos;
    out vec3 vs_eye_pos;
    out vec3 vs_light_eye_dir;
    out vec3 vs_normal;
    out float vs_eye_depth;

    void main() {
        gl_Position = u_mvp * vec4(in_position, 1.0);
        vs_color = in_color;
        vs_texcoord = in_texcoord;
        vs_obj_pos = in_position;
        vs_eye_pos = (u_mv * vec4(in_position, 1.0)).xyz; // Vertex position in eye frame.

        // OpenGL Z axis goes out of the screen, so depths are negative
        vs_eye_depth = -vs_eye_pos.z;

        vs_light_eye_dir = normalize(u_light_eye_pos - vs_eye_pos); // Vector to the light
        vs_normal = normalize(u_nm * vec4(in_normal, 1.0)).xyz; // Normal in eye frame.
    }
    """

    _fragment_shader = """
    uniform float u_ambient_coeff;
    uniform float u_diffuse_coeff;
    uniform float u_specular_coeff;
    uniform float u_specular_shininess;
    uniform float u_specular_whiteness;
    uniform sampler2D u_texture_map;
    uniform int u_use_texture;
    uniform float u_obj_id;
    uniform float u_instance_id;

    in vec3 vs_color;
    in vec2 vs_texcoord;
    in vec3 vs_obj_pos;
    in vec3 vs_eye_pos;
    in vec3 vs_light_eye_dir;
    in vec3 vs_normal;
    in float vs_eye_depth;

    layout(location = 0) out vec4 out_rgb;
    layout(location = 1) out vec4 out_depth;
    layout(location = 2) out vec4 out_seg;
    layout(location = 3) out vec4 out_instance_seg;
    layout(location = 4) out vec4 out_normal_map;
    layout(location = 5) out vec4 out_corr_map;

    void main() {
        float light_diffuse_w = max(dot(normalize(vs_light_eye_dir), normalize(vs_normal)), 0.0);

        vec3 cam_eye_dir = normalize(-vs_eye_pos); // Camera position term omitted. It is (0,0,0) in eye frame.
        vec3 reflect_dir = reflect(-normalize(vs_light_eye_dir), normalize(vs_normal));
        float light_specular_w = pow(max(dot(cam_eye_dir, reflect_dir), 0.0), u_specular_shininess);


        vec3 object_color;
        if(bool(u_use_texture))
            object_color = vec4(texture2D(u_texture_map, vs_texcoord)).rgb;
        else
            object_color = vs_color;

        vec3 specular_color = (1.0-u_specular_whiteness)*object_color + u_specular_whiteness*vec3(1.0, 1.0, 1.0);

        vec3 out_rgb_tmp = u_ambient_coeff*object_color + u_diffuse_coeff*light_diffuse_w*object_color + u_specular_coeff*light_specular_w*specular_color;
        out_rgb = vec4(out_rgb_tmp, 1.0);


        out_depth = vec4(vs_eye_depth, 0.0, 0.0, 1.0);
        out_seg = vec4(u_obj_id, 0.0, 0.0, 1.0);
        out_instance_seg = vec4(u_instance_id, 0.0, 0.0, 1.0);
        out_normal_map = vec4(vs_normal, 1.0);
        out_corr_map = vec4(vs_obj_pos, 1.0);
    }
    """

    def __init__(self, shape, coord_discretization_res = 1.0):
        self._shape = shape
        self._coord_discretization_res = coord_discretization_res
        self._mat_view = self._get_model_view_transf()
        self._vertex_buffers = {}
        self._index_buffers = {}
        self._texture_maps = {}
        self._window = app.Window(visible=False, width=self._shape[1], height=self._shape[0])
        self._program = self._setup_program()
        self._fbo = self._create_framebuffer()

    def __del__(self):
        self._window.close()

    def _get_model_view_transf(self):
        # View matrix (transforming also the coordinate system from OpenCV to
        # OpenGL camera frame)
        mat_view = np.eye(4, dtype=np.float32)
        mat_view[1, 1], mat_view[2, 2] = -1, -1
        mat_view = mat_view.T # OpenGL expects column-wise matrix format
        return mat_view

    # Functions to calculate transformation matrices
    # Note that OpenGL expects the matrices to be saved column-wise
    # (Ref: http://www.songho.ca/opengl/gl_transform.html)
    #-------------------------------------------------------------------------------
    # Model-view matrix
    def _compute_model_view(self, model, view):
        return np.dot(model, view)

    # Model-view-projection matrix
    def _compute_model_view_proj(self, model, view, proj):
        return np.dot(np.dot(model, view), proj)

    # Normal matrix (Ref: http://www.songho.ca/opengl/gl_normaltransform.html)
    def _compute_normal_matrix(self, model, view):
        return np.linalg.inv(np.dot(model, view)).T

    # Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix
    #-------------------------------------------------------------------------------
    # Ref:
    # 1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    # 2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
    def _compute_calib_proj(self, K, x0, y0, w, h, nc, fc, window_coords='y_down'):
        """
        :param K: Camera calibration matrix.
        :param x0, y0: The camera image origin (normally (0, 0)).
        :param w: Image width.
        :param h: Image height.
        :param nc: Near clipping plane.
        :param fc: Far clipping plane.
        :param window_coords: 'y_up' or 'y_down'.
        :return: OpenGL projection matrix.
        """
        depth = float(fc - nc)
        q = -(fc + nc) / depth
        qn = -2 * (fc * nc) / depth

        # Draw our images upside down, so that all the pixel-based coordinate
        # systems are the same
        if window_coords == 'y_up':
            proj = np.array([
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
                [0, 0, -1, 0]
            ]) # This row is also standard glPerspective

        # Draw the images right side up and modify the projection matrix so that OpenGL
        # will generate window coords that compensate for the flipped image coords
        else:
            assert window_coords == 'y_down'
            proj = np.array([
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
                [0, 0, -1, 0]
            ]) # This row is also standard glPerspective
        return proj.T

    def _setup_program(self, glsl_version='330'):
        program = gloo.Program(self._vertex_shader, self._fragment_shader, version=glsl_version)
        return program

    def _create_framebuffer(self):
        color_buf_rgb = np.empty((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        color_buf_depth = np.empty((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        color_buf_seg = np.empty((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        color_buf_instance_seg = np.empty((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        color_buf_normal_map = np.empty((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        color_buf_corr_map = np.empty((self._shape[0], self._shape[1], 4), np.float32).view(gloo.TextureFloat2D)

        depth_buf = np.empty((self._shape[0], self._shape[1]), np.float32).view(gloo.DepthTexture)

        fbo = gloo.FrameBuffer(
            color = [
                color_buf_rgb,
                color_buf_depth,
                color_buf_seg,
                color_buf_instance_seg,
                color_buf_normal_map,
                color_buf_corr_map,
            ],
            depth = depth_buf,
        )

        return fbo

    def _prepare_rendering(self):

        # OpenGL setup
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self._shape[1], self._shape[0])

        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        # gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
        # gl.glDisable(gl.GL_LINE_SMOOTH)
        # gl.glDisable(gl.GL_POLYGON_SMOOTH)
        # gl.glEnable(gl.GL_MULTISAMPLE)

        # Keep the back-face culling disabled because of objects which do not have
        # well-defined surface (e.g. the lamp from the dataset of Hinterstoisser)
        gl.glDisable(gl.GL_CULL_FACE)
        # gl.glEnable(gl.GL_CULL_FACE)
        # gl.glCullFace(gl.GL_BACK) # Back-facing polygons will be culled

    def _preprocess_object_model(self, obj_id, model, texture_map = None, surf_color = None):
        # Process input data
        #---------------------------------------------------------------------------
        # Make sure vertices and faces are provided in the model
        assert({'pts', 'faces'}.issubset(set(model.keys())))

        # Set texture / color of vertices
        if texture_map is not None:
            if texture_map.max() > 1.0:
                texture_map = texture_map.astype(np.float32) / 255.0
            texture_map = np.flipud(texture_map)
            texture_uv = model['texture_uv']
            colors = np.zeros((model['pts'].shape[0], 3), np.float32)
        else:
            texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)
            if not surf_color:
                if 'colors' in model.keys():
                    assert(model['pts'].shape[0] == model['colors'].shape[0])
                    colors = model['colors']
                    if colors.max() > 1.0:
                        colors /= 255.0 # Color values are expected in range [0, 1]
                else:
                    colors = np.ones((model['pts'].shape[0], 3), np.float32) * 0.5
            else:
                colors = np.tile(list(surf_color) + [1.0], [model['pts'].shape[0], 1])

        # Set the vertex data
        vertices_type = [('in_position', np.float32, 3),
                         ('in_normal', np.float32, 3),
                         ('in_color', np.float32, colors.shape[1]),
                         ('in_texcoord', np.float32, 2)]
        vertices = np.empty((model['pts'].shape[0],), vertices_type)
        vertices['in_position'] = model['pts']
        vertices['in_normal'] = model['normals']
        vertices['in_color'] = colors
        vertices['in_texcoord'] = texture_uv

        # Create buffers
        self._vertex_buffers[obj_id] = vertices.view(gloo.VertexBuffer)
        self._index_buffers[obj_id] = model['faces'].flatten().astype(np.uint32).view(gloo.IndexBuffer)
        self._texture_maps[obj_id] = texture_map

    # @profile
    def _draw(self, program, mat_proj, R, t, obj_id, instance_id):
        # Model matrix
        mat_model = np.eye(4, dtype=np.float32) # From world frame to eye frame
        mat_model[:3, :3], mat_model[:3, 3] = R, t.squeeze()
        mat_model = mat_model.T

        # Rendering
        self._program['u_mv'] = self._compute_model_view(mat_model, self._mat_view)
        self._program['u_nm'] = self._compute_normal_matrix(mat_model, self._mat_view)
        self._program['u_mvp'] = self._compute_model_view_proj(mat_model, self._mat_view, mat_proj)
        if self._texture_maps[obj_id] is not None:
            self._program['u_use_texture'] = int(True)
            self._program['u_texture_map'] = self._texture_maps[obj_id]
        else:
            self._program['u_use_texture'] = int(False)
            self._program['u_texture_map'] = np.zeros((1, 1, 4), np.float32)
        self._program['u_obj_id'] = obj_id
        self._program['u_instance_id'] = instance_id

        self._program.bind(self._vertex_buffers[obj_id])
        self._program.draw(gl.GL_TRIANGLES, self._index_buffers[obj_id])

    def _read_fbo(
        self,
        desired_buffers=['rgb', 'depth', 'seg', 'instance_seg', 'normal_map', 'corr_map'],
    ):
        def scale2uint8(array):
            return np.clip(array*256.0, 0.0, 255.999).astype(np.uint8)

        def discretize_and_convert_to_uint16(array):
            array /= self._coord_discretization_res
            # If coordinates are negative, subtract 1.0 in order to avoid rounding towards 0.
            array[array < 0.0] -= 1.0
            array = array.astype(np.uint16)
            return array

        buffers = {}

        if 'rgb' in desired_buffers:
            rgb = np.zeros((self._shape[0], self._shape[1], 4), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RGBA, gl.GL_FLOAT, rgb)
            rgb = np.flipud(rgb[:, :, :3])
            rgb = scale2uint8(rgb)
            buffers['rgb'] = rgb

        if 'depth' in desired_buffers:
            depth = np.zeros((self._shape[0], self._shape[1]), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
            gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RED, gl.GL_FLOAT, depth)
            depth = np.flipud(depth)
            depth = discretize_and_convert_to_uint16(depth)
            buffers['depth'] = depth

        if 'seg' in desired_buffers:
            seg = np.zeros((self._shape[0], self._shape[1]), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT2)
            gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RED, gl.GL_FLOAT, seg)
            seg = np.flipud(seg).astype(np.uint8)
            buffers['seg'] = seg

        if 'instance_seg' in desired_buffers:
            instance_seg = np.zeros((self._shape[0], self._shape[1]), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT3)
            gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RED, gl.GL_FLOAT, instance_seg)
            instance_seg = np.flipud(instance_seg).astype(np.uint8)
            buffers['instance_seg'] = instance_seg

        if 'normal_map' in desired_buffers:
            normal_map = np.zeros((self._shape[0], self._shape[1], 4), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT4)
            gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RGBA, gl.GL_FLOAT, normal_map)
            normal_map = np.flipud(normal_map[:, :, :3])
            normal_map = discretize_and_convert_to_uint16(normal_map)
            buffers['normal_map'] = normal_map

        if 'corr_map' in desired_buffers:
            corr_map = np.zeros((self._shape[0], self._shape[1], 4), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT5)
            gl.glReadPixels(0, 0, self._shape[1], self._shape[0], gl.GL_RGBA, gl.GL_FLOAT, corr_map)
            corr_map = np.flipud(corr_map[:, :, :3])
            corr_map = discretize_and_convert_to_uint16(corr_map)
            buffers['corr_map'] = corr_map

        return buffers

    # @profile
    def render(
        self,
        K,
        R_list,
        t_list,
        obj_id_list,
        light_pos = [0, 0, 0], # Camera origin
        ambient_weight = 0.5,
        diffuse_weight = 0.5,
        specular_weight = 0.0,
        specular_shininess = 3.,
        specular_whiteness = 0.3,
        clip_near = 100,
        clip_far = 10000,
        desired_buffers = ['rgb', 'depth', 'seg', 'instance_seg', 'normal_map', 'corr_map'],
    ):
        assert ambient_weight + diffuse_weight <= 1.0 + 1e-7
        assert specular_shininess >= 0.0
        nbr_instances = len(R_list)

        mat_proj = self._compute_calib_proj(K, 0, 0, self._shape[1], self._shape[0], clip_near, clip_far)

        light_pos = np.concatenate([np.array(light_pos).squeeze(), [1.]])
        light_pos = np.squeeze(self._mat_view @ light_pos) # self._mat_view represents cam frame -> openGL cam frame transformation. (Camera assumed to be in origin)
        light_pos = light_pos[:3] / light_pos[3] # Perspective divide
        self._program['u_light_eye_pos'] = light_pos
        self._program['u_ambient_coeff'] = ambient_weight
        self._program['u_diffuse_coeff'] = diffuse_weight
        self._program['u_specular_coeff'] = specular_weight
        self._program['u_specular_shininess'] = specular_shininess
        self._program['u_specular_whiteness'] = specular_whiteness

        self._fbo.activate()
        self._prepare_rendering() # Could alternatively be done in on_draw()

        @self._window.event
        def on_draw(dt):
            # self._window.clear() # Instead of gl.glClear(...)
            instance_id = 0
            for R, t, obj_id in zip(R_list, t_list, obj_id_list):
                instance_id += 1
                self._draw(self._program, mat_proj, R, t, obj_id, instance_id)

        app.run(framecount=0) # The on_draw function is called framecount+1 times

        buffers = self._read_fbo(desired_buffers = desired_buffers)
        self._fbo.deactivate()

        return buffers

from lib.constants import TRAIN, VAL, TEST
from lib.rendering.neural_rendering_wrapper import NeuralRenderingWrapper
from lib.rendering.glumpy_rendering_wrapper import GlumpyRenderingWrapper

global global_neural_renderer
global_neural_renderer = None
def get_neural_renderer(configs):
    global global_neural_renderer
    if global_neural_renderer is None:
        global_neural_renderer = NeuralRenderingWrapper(configs)
    return global_neural_renderer

global global_glumpy_renderer
global_glumpy_renderer = None
def get_glumpy_renderer(configs):
    global global_glumpy_renderer
    if global_glumpy_renderer is None:
        lowres_render_size = configs.data.query_rendering_opts.lowres_render_size
        if lowres_render_size is None:
            lowres_render_size = configs.data.crop_dims
        modes = (TRAIN, VAL) if configs.train_or_eval == 'train' else (TEST,)
        if any([ ref_scheme['ref_source'] == 'synthetic' for mode in modes for schemset_name, schemeset_def in configs.runtime.ref_sampling_schemes[mode].items() for ref_scheme in schemeset_def ]):
            max_render_dims = (
                max(configs.data.img_dims[0], lowres_render_size[0]),
                max(configs.data.img_dims[1], lowres_render_size[1]),
            )
        else:
            max_render_dims = lowres_render_size
        global_glumpy_renderer = GlumpyRenderingWrapper(
            configs,
            max_render_dims = max_render_dims,
        )
    return global_glumpy_renderer

def get_ref_renderer(configs):
    return get_glumpy_renderer(configs)

def get_query_renderer(configs):
    if configs.data.query_rendering_method == 'neural':
        return get_neural_renderer(configs)
    elif configs.data.query_rendering_method == 'glumpy':
        return get_glumpy_renderer(configs)
    assert False

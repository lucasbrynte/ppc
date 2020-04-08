from lib.rendering.neural_rendering_wrapper import NeuralRenderingWrapper

global global_renderer
global_renderer = None
def get_renderer(configs):
    global global_renderer
    if global_renderer is None:
        if self._configs.data.query_rendering_method == 'neural':
            global_renderer = NeuralRenderingWrapper(configs)
        else:
            assert False
    return global_renderer

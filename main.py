import torch
from torch import nn
import numpy as np
from attrdict import AttrDict
from importlib import import_module
import time

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL, TEST, CONFIG_ROOT
from lib.loss import LossHandler
from lib.pose_optimizer import FullPosePipeline, PoseOptimizer
from lib.utils import get_device, get_module_parameters
from lib.rendering.renderer_interface import get_query_renderer
from lib.utils import crop_and_rescale_pt_batched
from lib.visualize import Visualizer
from lib.loader import Loader


# class GELU(nn.Module):
#     r"""Applies the Gaussian Error Linear Units function:
# 
#     .. math::
#         \text{GELU}(x) = x * \Phi(x)
#     where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
# 
#     Shape:
#         - Input: :math:`(N, *)` where `*` means, any number of additional
#           dimensions
#         - Output: :math:`(N, *)`, same shape as the input
# 
#     .. image:: scripts/activation_images/GELU.png
# 
#     Examples::
# 
#         >>> m = nn.GELU()
#         >>> input = torch.randn(2)
#         >>> output = m(input)
#     """
#     def forward(self, input):
#         return nn.functional.gelu(input)
# 
# def replace_relu_activations(model, new_activation):
#     """
#     https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7
#     """
#     for child_name, child in model.named_children():
#         if isinstance(child, nn.ReLU):
#             setattr(model, child_name, new_activation())
#         else:
#             replace_relu_activations(child, new_activation)

class Main():
    def __init__(self, configs):
        """Constructor."""
        self._configs = configs
        self._model_module = import_module('lib.models.%s' % configs.model.architecture)
        self._modes = (TRAIN, VAL) if self._configs.train_or_eval == 'train' else (TEST,)
        self._data_loader = Loader(self._modes, self._configs)
        self._loss_handler = LossHandler(configs, self.__class__.__name__)
        self._checkpoint_handler = CheckpointHandler(configs)
        self._model = self._checkpoint_handler.init(self._init_model())
        if self._configs.train_or_eval == 'train':
            self._optimizer = self._setup_optimizer()
            # self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='min', factor=0.3)
            self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, 10, gamma=0.3) # Multiply by 0.3 every 10 epochs.
        self._visualizer = Visualizer(configs)

        self._query_renderer = get_query_renderer(configs)

        self._target_prior_samples_numpy = None
        self._target_prior_samples = None

    def _init_model(self):
        model = self._model_module.Model(self._configs)
        # replace_relu_activations(model, GELU)
        # replace_relu_activations(model, nn.Softplus)
        return model

    def _setup_optimizer(self):
        param_groups = self._model.get_optim_param_groups()
        return torch.optim.Adam(
            param_groups,
            lr=self._configs.training.learning_rate * np.sqrt(self._configs.runtime.data_sampling_scheme_defs.train.train.opts.loading.batch_size),
        )

    def train(self):
        if any([task_spec['prior_loss'] is not None for task_name, task_spec in self._configs.tasks.items()]):
            target_prior_samples = self._sample_epoch_of_targets(TRAIN, TRAIN)
            self._target_prior_samples_numpy = target_prior_samples
            self._target_prior_samples = {task_name: torch.tensor(target_prior_samples[task_name], device=get_device()).float() for task_name in target_prior_samples.keys()}

        for epoch in range(1, self._configs.training.n_epochs + 1):

            train_score = -self._run_epoch(epoch,
                TRAIN,
                TRAIN,
                save_imgs_flag = self._configs.runtime.data_sampling_scheme_defs[TRAIN][TRAIN]['opts']['visualization']['save_imgs_interval'] is not None and epoch % self._configs.runtime.data_sampling_scheme_defs[TRAIN][TRAIN]['opts']['visualization']['save_imgs_interval'] == 0,
                plot_signals_flag = self._configs.runtime.data_sampling_scheme_defs[TRAIN][TRAIN]['opts']['visualization']['plot_signals_interval'] is not None and epoch % self._configs.runtime.data_sampling_scheme_defs[TRAIN][TRAIN]['opts']['visualization']['plot_signals_interval'] == 0,
                plot_signal_stats_flag = self._configs.runtime.data_sampling_scheme_defs[TRAIN][TRAIN]['opts']['visualization']['plot_signal_stats_interval'] is not None and epoch % self._configs.runtime.data_sampling_scheme_defs[TRAIN][TRAIN]['opts']['visualization']['plot_signal_stats_interval'] == 0,
            )

            val_scores = {}
            for schemeset in self._configs.runtime.data_sampling_scheme_defs[VAL].keys():
                with torch.no_grad():
                    score = -self._run_epoch(epoch,
                        VAL,
                        schemeset,
                        save_imgs_flag = self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['opts']['visualization']['save_imgs_interval'] is not None and epoch % self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['opts']['visualization']['save_imgs_interval'] == 0,
                        plot_signals_flag = self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['opts']['visualization']['plot_signals_interval'] is not None and epoch % self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['opts']['visualization']['plot_signals_interval'] == 0,
                        plot_signal_stats_flag = self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['opts']['visualization']['plot_signal_stats_interval'] is not None and epoch % self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['opts']['visualization']['plot_signal_stats_interval'] == 0,
                    )
                if self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['use_for_val_score']:
                    val_scores[schemeset] = score
            val_score = sum(val_scores.values()) / len(val_scores) if len(val_scores) > 0 else None

            # self._lr_scheduler.step(val_score, epoch=None) # Make sure that the val_score argument is not confused for the epoch.
            self._lr_scheduler.step(epoch=None)
            # self._checkpoint_handler.save(self._model, epoch, train_score)
            self._checkpoint_handler.save(self._model, epoch, score=val_score)

    def eval(self):
        epoch = 1
        for schemeset in sorted(
            self._configs.runtime.data_sampling_scheme_defs[TEST].keys(),
            key = lambda schemeset: self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['loading']['nbr_batches'],
        ):
        # for schemeset in self._configs.runtime.data_sampling_scheme_defs[TEST]:
            with torch.no_grad():
                self._run_epoch(
                    epoch,
                    TEST,
                    schemeset,
                    save_imgs_flag = self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['visualization']['save_imgs'],
                    plot_signals_flag = self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['visualization']['plot_signals'],
                    plot_signal_stats_flag = self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['visualization']['plot_signal_stats'],
                )

    def eval_with_posesearch(self):
        for schemeset, schemeset_def in self._configs.runtime.data_sampling_scheme_defs[TEST].items():
            if not self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['poseopt']['enabled']:
                continue
            self._run_epoch_with_posesearch(
                TEST,
                schemeset,
                self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['poseopt']['mode'],
            )

    def _sample_epoch_of_targets(self, mode, schemeset):
        print('Running through epoch to collect target samples for prior...')
        selected_targets = {task_spec['target'] for task_name, task_spec in self._configs.tasks.items() if task_spec['prior_loss'] is not None}

        cnt = 1
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode, schemeset, self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['nbr_batches'] * self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['batch_size'])):
            pertarget_target_features = self._loss_handler.get_target_features(batch.targets, selected_targets=selected_targets)
            # Map target features to corresponding tasks:
            target_features = self._loss_handler.map_features_to_tasks(pertarget_target_features)
            target_features_raw = self._loss_handler.apply_inverse_activation(target_features)
            self._loss_handler.record_batch_of_persample_signals('target_feat_raw', target_features_raw)
            if cnt % 10 == 0:
                print('{}/{}'.format(cnt, self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['nbr_batches']))
            cnt += 1

        target_samples = self._loss_handler.get_persample_signals_numpy()['target_feat_raw']
        self._loss_handler._reset_signals()
        print('Done.')
        return target_samples

    def _run_epoch_with_posesearch(self, mode, schemeset, poseopt_mode):
        assert mode == TEST
        self._model.eval() # Although gradients are computed - batch norm, dropout etc should be in eval mode.
        numerical_grad = True
        # numerical_grad = False
        with torch.set_grad_enabled(not numerical_grad):
            # NOTE: This is how to enable deterministic ref scheme sampling
            nbr_batches = None
            # nbr_batches = 100 # hack for being able to evaluate on synth
            for batch_id, batch in enumerate(self._data_loader.gen_batches(mode, schemeset, nbr_batches)):
                batch_size = self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['batch_size']
                # if self._configs.obj_label == 'cat' and batch_size*batch_id < 67:
                #     print('Skipping batch {}'.format(batch_size * batch_id))
                #     continue
                print('{} samples done...'.format(batch_size * batch_id))
                maps, extra_input = self._batch_to_gpu(batch.maps, batch.extra_input)
                pose_pipeline = FullPosePipeline(
                    self._configs,
                    self._model,
                    self._query_renderer,
                    maps.ref_img_full,
                    extra_input.K,
                    extra_input.obj_diameter,
                    batch.meta_data.obj_id,
                    maps.ref_img_full.shape[0]*[self._configs.data.query_rendering_opts.ambient_weight],
                )
                def nn_out2interp_pred_features(nn_out):
                    pred_features_raw = self._loss_handler.get_pred_features(nn_out['features'])
                    pred_features = self._loss_handler.apply_activation(pred_features_raw)
                    interp_pred_features = self._loss_handler.calc_human_interpretable_features(pred_features)
                    return interp_pred_features
                pose_optimizer = PoseOptimizer(
                    self._configs,
                    pose_pipeline,
                    nn_out2interp_pred_features,
                    extra_input.K,
                    extra_input.R1, # R_gt
                    extra_input.t1, # t_gt
                    extra_input.R1, # R_refpt
                    batch.meta_data.ref_img_path,
                    numerical_grad = numerical_grad,
                )

                if poseopt_mode == 'optimize_all':
                    optimize = True
                    opt_enable_plotting = False
                    opt_print_iterates = False
                    evaluate_and_plot = False
                    break_after_first_batch = False
                    N = 100
                elif poseopt_mode == 'optimize_one_frame':
                    optimize = True
                    opt_enable_plotting = True
                    opt_print_iterates = True
                    evaluate_and_plot = False
                    break_after_first_batch = True
                    N = 100
                elif poseopt_mode == 'eval_init':
                    optimize = True
                    opt_enable_plotting = False
                    opt_print_iterates = False
                    evaluate_and_plot = False
                    break_after_first_batch = False
                    N = 0
                elif poseopt_mode == 'eval_init_and_render_one_frame':
                    optimize = True
                    opt_enable_plotting = True
                    opt_print_iterates = False
                    evaluate_and_plot = False
                    break_after_first_batch = True
                    N = 0
                elif poseopt_mode == 'evaluate_and_plot_one_frame':
                    optimize = False
                    evaluate_and_plot = True
                    break_after_first_batch = True
                else:
                    assert False

                if optimize:
                    init_pose = {'init': (extra_input.R2_init, extra_input.t2_init)}
                    # init_pose = {'init': (extra_input.R1, extra_input.t1)}
                    if len(self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['poseopt']['aug_init_pose_zrot_objects']) > 0:
                        if self._configs.obj_label in self._configs.runtime.data_sampling_scheme_defs[TEST][schemeset]['opts']['poseopt']['aug_init_pose_zrot_objects']:
                            R0, t0 = init_pose['init']
                            R0 = R0.clone()
                            R0[:,:,:2] *= -1. # Change sign of first two columns - corresponding to 180 degree rotaiton around z-axis in object frame.
                            init_pose['zrot180'] = R0, t0
                    t0 = time.time()
                    pose_optimizer.optimize(
                        init_pose,
                        N = N,
                    
                        # optim_runs = {
                        #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # },
                        # 
                        # optim_runs = {
                        #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_plus_x': {'deg_perturb': 20., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_minus_x': {'deg_perturb': 20., 'axis_perturb': [-1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_plus_y': {'deg_perturb': 20., 'axis_perturb': [0., 1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_minus_y': {'deg_perturb': 20., 'axis_perturb': [0., -1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_plus_z': {'deg_perturb': 20., 'axis_perturb': [0., 0., 1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_minus_z': {'deg_perturb': 20., 'axis_perturb': [0., 0., -1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # },
                    
                        # optim_runs = {
                        #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'tetra_perturb1': {'deg_perturb': 20., 'axis_perturb': [1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'tetra_perturb2': {'deg_perturb': 20., 'axis_perturb': [-1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'tetra_perturb3': {'deg_perturb': 20., 'axis_perturb': [0., 1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'tetra_perturb4': {'deg_perturb': 20., 'axis_perturb': [0., -1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # },
                        # 
                        # num_wxdims = 3,
                        # num_txdims = 0,
                        # num_ddims = 0,
                    
                        # # optim_runs = {
                        # #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # # },
                        # 
                        # optim_runs = {
                        #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_plus_x': {'deg_perturb': 20., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_minus_x': {'deg_perturb': 20., 'axis_perturb': [-1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_plus_y': {'deg_perturb': 20., 'axis_perturb': [0., 1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_minus_y': {'deg_perturb': 20., 'axis_perturb': [0., -1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_plus_z': {'deg_perturb': 20., 'axis_perturb': [0., 0., 1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'init_gt_minus_z': {'deg_perturb': 20., 'axis_perturb': [0., 0., -1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # },
                        # 
                        # # optim_runs = {
                        # #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'tetra_perturb1': {'deg_perturb': 20., 'axis_perturb': [1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'tetra_perturb2': {'deg_perturb': 20., 'axis_perturb': [-1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'tetra_perturb3': {'deg_perturb': 20., 'axis_perturb': [0., 1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'tetra_perturb4': {'deg_perturb': 20., 'axis_perturb': [0., -1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # # },
                        # 
                        # num_wxdims = 3,
                        # num_txdims = 2,
                        # num_ddims = 1,
                    
                        # # optim_runs = {
                        # #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # # },
                        # 
                        # # optim_runs = {
                        # #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_plus_x': {'deg_perturb': 20., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_minus_x': {'deg_perturb': 20., 'axis_perturb': [-1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_plus_y': {'deg_perturb': 20., 'axis_perturb': [0., 1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_minus_y': {'deg_perturb': 20., 'axis_perturb': [0., -1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_plus_z': {'deg_perturb': 20., 'axis_perturb': [0., 0., 1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_minus_z': {'deg_perturb': 20., 'axis_perturb': [0., 0., -1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # # },
                        # 
                    
                    
                    
                    
                    
                        # 6D
                        optim_runs = {
                            'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.0},
                            # 'tetra_perturb1': {'deg_perturb': 20., 'axis_perturb': [1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [30., -0.], 'd_perturb': 0.7},
                            # 'tetra_perturb2': {'deg_perturb': 20., 'axis_perturb': [-1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., -30.], 'd_perturb': 1.1},
                            # 'tetra_perturb3': {'deg_perturb': 20., 'axis_perturb': [0., 1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 30.], 'd_perturb': 1.3},
                            # 'tetra_perturb4': {'deg_perturb': 20., 'axis_perturb': [0., -1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [-30., 0.], 'd_perturb': 0.8},
                        },
                    
                        num_wxdims = 3,
                        num_txdims = 2,
                        num_ddims = 1,
                    
                        # 5D
                        # optim_runs = {
                        #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'tetra_perturb1': {'deg_perturb': 20., 'axis_perturb': [1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [30., -0.], 'd_perturb': 1.},
                        #     'tetra_perturb2': {'deg_perturb': 20., 'axis_perturb': [-1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., -30.], 'd_perturb': 1.},
                        #     'tetra_perturb3': {'deg_perturb': 20., 'axis_perturb': [0., 1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 30.], 'd_perturb': 1.},
                        #     'tetra_perturb4': {'deg_perturb': 20., 'axis_perturb': [0., -1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [-30., 0.], 'd_perturb': 1.},
                        # },
                        # 
                        # num_wxdims = 3,
                        # num_txdims = 2,
                        # num_ddims = 0,
                    
                    
                        # # optim_runs = {
                        # #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # # },
                        # 
                        # # optim_runs = {
                        # #     'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_plus_x': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_minus_x': {'deg_perturb': 0., 'axis_perturb': [-1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_plus_y': {'deg_perturb': 0., 'axis_perturb': [0., 1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_minus_y': {'deg_perturb': 0., 'axis_perturb': [0., -1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_plus_z': {'deg_perturb': 0., 'axis_perturb': [0., 0., 1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # #     'init_gt_minus_z': {'deg_perturb': 0., 'axis_perturb': [0., 0., -1.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # # },
                        # 
                        # optim_runs = {
                        #     # 'noperturb': {'deg_perturb': 0., 'axis_perturb': [1., 0., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     'tetra_perturb1': {'deg_perturb': 0., 'axis_perturb': [1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     # 'tetra_perturb2': {'deg_perturb': 0., 'axis_perturb': [-1., 0., -1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     # 'tetra_perturb3': {'deg_perturb': 0., 'axis_perturb': [0., 1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        #     # 'tetra_perturb4': {'deg_perturb': 0., 'axis_perturb': [0., -1., 1./np.sqrt(2)], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.], 'd_perturb': 1.},
                        # },
                        # 
                        # num_wxdims = 0,
                        # num_txdims = 2,
                        # num_ddims = 0,
                    
                        enable_plotting = opt_enable_plotting,
                        print_iterates = opt_print_iterates,
                        store_eval = True,
                    )
                    full_optim_time = time.time() - t0
                    if N > 0:
                        print('full time: {} s, time / sample: {} s, time / sample and iter: {} s'.format(
                            full_optim_time,
                            full_optim_time / batch_size,
                            (full_optim_time / batch_size) / N,
                        ))
                if evaluate_and_plot:
                    pose_optimizer.evaluate(
                        num_wxdims = 1,
                        # num_txdims = 2,
                        # num_ddims = 0,
                        # N_each = [2, 10],
                        # num_txdims = 2,
                        # num_ddims = 0,
                        # N_each = [20, 20],
                        num_txdims = 0,
                        num_ddims = 0,
                        N_each = [101],
                        # N_each = [11],
                        # primary_w_dir = None,
                        primary_w_dir = [1.0, 1.0, 1.0],
                        calc_grad=False,
                        # calc_grad=True,
                        store_eval = True,
                    )
                if break_after_first_batch:
                    break

    def _run_epoch(
            self,
            epoch,
            mode,
            schemeset,
            save_imgs_flag=True,
            plot_signals_flag=True,
            plot_signal_stats_flag=True,
        ):
        if mode == TRAIN:
            self._model.train()
            if self._configs.model.architecture == 'resnet18' and self._configs.model.resnet18_opts.encoder_freeze_nbr_epochs is not None:
                if epoch <= self._configs.model.resnet18_opts.encoder_freeze_nbr_epochs:
                    self._model.freeze_encoder()
                else:
                    self._model.unfreeze_encoder()
        else:
            self._model.eval()

        bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # cnt = 0
        visual_cnt = 1
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode, schemeset, self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['nbr_batches'] * self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['batch_size'])):
            maps, extra_input = self._batch_to_gpu(batch.maps, batch.extra_input)

            pose_pipeline = FullPosePipeline(
                self._configs,
                self._model,
                self._query_renderer,
                maps.ref_img_full,
                extra_input.K,
                extra_input.obj_diameter,
                batch.meta_data.obj_id,
                maps.ref_img_full.shape[0]*[self._configs.data.query_rendering_opts.ambient_weight],
            )
            H, ref_img, query_img, nn_out = pose_pipeline(extra_input.R2, extra_input.t2)
            instance_seg1 = crop_and_rescale_pt_batched(maps.instance_seg1_full, H, self._configs.data.crop_dims, interpolation_mode='nearest')
            safe_fg_anno_mask = crop_and_rescale_pt_batched(maps.safe_fg_anno_mask_full, H, self._configs.data.crop_dims, interpolation_mode='nearest')

            # Raw predicted features (neural net output)
            pred_features_raw = self._loss_handler.get_pred_features(nn_out['features'])

            # Apply activation, and get corresponding target features
            pred_features = self._loss_handler.apply_activation(pred_features_raw)
            pertarget_target_features = self._loss_handler.get_target_features(batch.targets)
            # Map target features to corresponding tasks:
            target_features = self._loss_handler.map_features_to_tasks(pertarget_target_features)

            if self._configs.training.clamp_predictions:
                # Clamp features before loss computation (for the features where desired)
                pred_features = self._loss_handler.clamp_features(pred_features, before_loss=True)

            # Calculate loss
            task_loss_decays, loss_notapplied = self._loss_handler.calc_loss_decay(target_features, pertarget_target_features, tasks_punished=batch.meta_data.tasks_punished, ref_scheme_loss_weight=extra_input.ref_scheme_loss_weight, query_scheme_loss_weight=extra_input.query_scheme_loss_weight)
            if mode in (TRAIN, VAL):
                task_losses = self._loss_handler.calc_loss(pred_features, target_features, task_loss_decays)
                if 'fg_mask' in nn_out and self._configs.aux_tasks.fg_mask.enabled:
                    fg_mask_loss = bce_loss(nn_out['fg_mask'], (instance_seg1 == 1).float())

                    if self._configs.aux_tasks.fg_mask.anno_trust_mode == 'everywhere':
                        # Trust segmentation everywhere
                        fg_mask_loss = fg_mask_loss.mean()
                    elif self._configs.aux_tasks.fg_mask.anno_trust_mode == 'matching_depth':
                        # # Only trust segmentation at matching depths. Effectively disregards all BG though, since rendered depth is zero...
                        fg_mask_loss = fg_mask_loss[safe_fg_anno_mask].mean()
                    elif self._configs.aux_tasks.fg_mask.anno_trust_mode == 'bg_or_matching_depth':
                        # # Always trust BG anno, but only trust FG anno, if depth also matches.
                        fg_mask_loss = fg_mask_loss[(instance_seg1 != 1) | safe_fg_anno_mask].mean()
                    else:
                        assert False

                    # print(loss, fg_mask_loss)
                    task_losses['fg_mask'] = 0.3 * fg_mask_loss
                if any([task_spec['prior_loss'] is not None for task_name, task_spec in self._configs.tasks.items()]):
                    prior_loss_signal_vals = self._loss_handler.calc_prior_loss(pred_features_raw, self._target_prior_samples)
                    task_losses['prior'] = sum(prior_loss_signal_vals.values())
                loss = sum(task_losses.values())

            if self._configs.training.clamp_predictions:
                # Clamp features after loss computation (for all features)
                pred_features = self._loss_handler.clamp_features(pred_features, before_loss=False)

            # Map features to interpretable domain (degrees etc.)
            interp_pred_features = self._loss_handler.calc_human_interpretable_features(pred_features)
            interp_target_features = self._loss_handler.calc_human_interpretable_features(target_features)

            # Feature errors
            interp_feat_error = self._loss_handler.calc_feature_errors(interp_pred_features, interp_target_features)
            interp_feat_abserror = self._loss_handler.calc_norm(interp_feat_error)
            interp_target_feat_norm = self._loss_handler.calc_norm(interp_target_features)
            relative_feat_abserror = self._loss_handler.normalize_interp_vals(interp_feat_abserror, interp_target_feat_norm)

            # Per-batch signals - will be plotted against epoch in TensorBoard
            if mode in (TRAIN, VAL):
                self._loss_handler.record_batch_of_perbatch_signals('loss', {'loss': loss})
                self._loss_handler.record_batch_of_perbatch_signals('task_losses', task_losses)
                if any([task_spec['prior_loss'] is not None for task_name, task_spec in self._configs.tasks.items()]):
                    self._loss_handler.record_batch_of_perbatch_signals('prior_losses', prior_loss_signal_vals)
                self._loss_handler.record_batch_of_perbatch_signals('relative_feat_abserror_avg', self._loss_handler.calc_batch_signal_avg(relative_feat_abserror))
                self._loss_handler.record_batch_of_perbatch_signals('interp_feat_abserror_avg', self._loss_handler.calc_batch_signal_avg(interp_feat_abserror))
                self._loss_handler.record_batch_of_perbatch_signals('relative_feat_abserror_avg_filtered', self._loss_handler.calc_batch_signal_avg(relative_feat_abserror, discard_signal=loss_notapplied))
                self._loss_handler.record_batch_of_perbatch_signals('interp_feat_abserror_avg_filtered', self._loss_handler.calc_batch_signal_avg(interp_feat_abserror, discard_signal=loss_notapplied))
            if mode == TRAIN:
                self._loss_handler.record_batch_of_perbatch_signals('lr', {'lr': torch.tensor([ group['lr'] for group in self._optimizer.param_groups ]).mean()})

            # Per-sample signals, e.g. feature values & corresponding errors
            self._loss_handler.record_batch_of_persample_signals('loss_notapplied', loss_notapplied)
            self._loss_handler.record_batch_of_persample_signals('relative_feat_abserror', relative_feat_abserror)
            self._loss_handler.record_batch_of_persample_signals('interp_feat_abserror', interp_feat_abserror)
            self._loss_handler.record_batch_of_persample_signals('interp_feat_error', interp_feat_error)
            self._loss_handler.record_batch_of_persample_signals('interp_pred_feat', interp_pred_features)
            self._loss_handler.record_batch_of_persample_signals('interp_target_feat', interp_target_features)
            self._loss_handler.record_batch_of_persample_signals('pred_feat', pred_features)
            self._loss_handler.record_batch_of_persample_signals('target_feat', target_features)
            self._loss_handler.record_batch_of_persample_signals('pred_feat_raw', pred_features_raw)

            if mode == TRAIN:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            self._loss_handler.log_batch(epoch, batch_id, mode, schemeset)

            # for task_name in sorted(self._configs.tasks.keys()):
            #     tmp = np.sqrt(target_features[task_name].detach().cpu().numpy())
            #     print('{} - batch std: {}'.format(task_name, np.sqrt(np.mean(tmp**2))))
            #     print('{} - batch median energy: {}'.format(task_name, np.sqrt(np.median(tmp**2))))

            # cnt += 1
            # if cnt % 10 == 0:
            #     self._visualizer.report_signals(self._loss_handler.get_scalar_averages(), mode)

            if mode == TEST:
                if save_imgs_flag:
                    print('Saving images...')
                    for sample_idx in range(self._configs.runtime.data_sampling_scheme_defs[mode][schemeset]['opts']['loading']['batch_size']):
                        self._visualizer.save_images(ref_img.cpu(), query_img.cpu(), batch.meta_data.ref_img_path, pred_features, target_features, loss_notapplied, mode, schemeset, visual_cnt, sample=sample_idx)
                        visual_cnt += 1

        if save_imgs_flag:
            print('Saving images...')
            if mode in (TRAIN, VAL):
                self._visualizer.save_images(ref_img.cpu(), query_img.cpu(), batch.meta_data.ref_img_path, pred_features, target_features, loss_notapplied, mode, schemeset, epoch, sample=-1)

        if plot_signals_flag:
            print('Plotting signals...')
            if mode in (TRAIN, VAL):
                self._visualizer.report_perbatch_signals(self._loss_handler.get_scalar_averages(), mode, schemeset, epoch)

        if plot_signal_stats_flag:
            print('Filtering samples where loss not applied...')
            self._loss_handler.filter_persample_signals([
                'interp_target_feat',
                'interp_pred_feat',
                'pred_feat',
                'pred_feat_raw',
                'relative_feat_abserror',
                'interp_feat_abserror',
                'interp_feat_error',
            ])
            print('Plotting signal stats...')
            # for filtered_flag in [False]:
            for filtered_flag in [True]:
            # for filtered_flag in [False, True]:
                self._visualizer.calc_and_plot_signal_stats(self._loss_handler.get_persample_signals_numpy(), mode, schemeset, epoch, target_prior_samples=self._target_prior_samples_numpy, filtered_flag=filtered_flag)
        print('Visualization done.')

        # for task_name in sorted(self._configs.tasks.keys()):
        #     tmp = np.sqrt(self._loss_handler.get_persample_signals_numpy()['target_feat'][task_name])
        #     print('{} - global std: {}'.format(task_name, np.sqrt(np.mean(tmp**2))))
        #     print('{} - global median energy: {}'.format(task_name, np.sqrt(np.median(tmp**2))))
        # assert False

        score = self._loss_handler.get_scalar_averages()['loss']['loss'] if mode in (TRAIN, VAL) else None
        self._loss_handler._reset_signals()
        return score

    def _batch_to_gpu(self, maps, extra_input):
        maps = maps.__class__(*tuple(map(lambda x: x.to(get_device(), non_blocking=True), maps)))
        extra_input = extra_input.__class__(*tuple(map(lambda x: x.to(get_device(), non_blocking=True), extra_input)))
        return maps, extra_input


def run(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, args.train_or_eval)
    setup.prepare_environment()
    setup.save_settings(args)
    configs = setup.get_configs(args)

    main = Main(configs)
    # configs['data']['data_loader'] = main._data_loader
    if args.train_or_eval == 'train':
        main.train()
    elif args.train_or_eval == 'eval':
        main.eval()
    else:
        assert args.train_or_eval == 'eval_poseopt'
        main.eval_with_posesearch()

if __name__ == '__main__':
    run(lib.setup)

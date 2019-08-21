import torch
import numpy as np
from numbers import Number
from attrdict import AttrDict

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL, TEST, CONFIG_ROOT
from lib.loss import LossHandler
from lib.model import Model
from lib.utils import get_device, get_module_parameters
from lib.visualize import Visualizer
from lib.loader import Loader


class Main():
    def __init__(self, configs):
        """Constructor."""
        self._configs = configs
        self._modes = (TRAIN, VAL) if self._configs.train_or_eval == 'train' else (TEST,)
        self._data_loader = Loader(self._modes, self._configs)
        self._loss_handler = LossHandler(configs, self.__class__.__name__)
        self._checkpoint_handler = CheckpointHandler(configs)
        self._model = self._checkpoint_handler.init(self._init_model())
        if self._configs.train_or_eval == 'train':
            self._optimizer = self._setup_optimizer()
            self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max')
        self._visualizer = Visualizer(configs)

        self._target_prior_samples_numpy = None
        self._target_prior_samples = None

    def _init_model(self):
        model = Model(self._configs)
        return model

    def _setup_optimizer(self):
        weight_decay = self._configs.training.weight_decay if self._configs.training.weight_decay is not None else {'cnn': 0, 'head': 0}
        assert all(isinstance(val, Number) for val in weight_decay.values())
        param_groups = []
        param_groups.append({
            'params': [param for name, param in self._model.semi_siamese_cnn.named_parameters() if name.endswith('weight') and param.requires_grad],
            'weight_decay': self._configs.training.weight_decay['cnn'],
        })
        param_groups.append({
            'params': [param for name, param in self._model.semi_siamese_cnn.named_parameters() if name.endswith('bias') and param.requires_grad],
            'weight_decay': 0,
        })
        param_groups.append({
            'params': [param for name, param in self._model.head.named_parameters() if name.endswith('weight') and param.requires_grad],
            'weight_decay': self._configs.training.weight_decay['head'],
        })
        param_groups.append({
            'params': [param for name, param in self._model.head.named_parameters() if name.endswith('bias') and param.requires_grad],
            'weight_decay': 0,
        })
        nbr_params = sum([len(param_group['params']) for param_group in param_groups])
        total_nbr_params = len([param for param in self._model.parameters() if param.requires_grad])
        assert nbr_params == total_nbr_params
        return torch.optim.Adam(
            param_groups,
            lr=self._configs.training.learning_rate * np.sqrt(self._configs.runtime.loading.batch_size),
        )

    def train(self):
        if any([task_spec['prior_loss'] is not None for task_name, task_spec in self._configs.tasks.items()]):
            target_prior_samples = self._sample_epoch_of_targets(TRAIN, TRAIN, self._configs.runtime.loading.nbr_batches_prior)
            self._target_prior_samples_numpy = target_prior_samples
            self._target_prior_samples = {task_name: torch.tensor(target_prior_samples[task_name], device=get_device()).float() for task_name in target_prior_samples.keys()}

        for epoch in range(1, self._configs.training.n_epochs + 1):
            save_imgs_flag = self._configs.runtime.visualization.save_imgs_interval is not None and epoch % self._configs.runtime.visualization.save_imgs_interval == 0
            plot_signals_flag = self._configs.runtime.visualization.plot_signals_interval is not None and epoch % self._configs.runtime.visualization.plot_signals_interval == 0
            plot_signal_stats_flag = self._configs.runtime.visualization.plot_signal_stats_interval is not None and epoch % self._configs.runtime.visualization.plot_signal_stats_interval == 0

            train_score = -self._run_epoch(epoch, TRAIN, TRAIN, self._configs.runtime.loading.nbr_batches_train, save_imgs_flag=save_imgs_flag, plot_signals_flag=plot_signals_flag, plot_signal_stats_flag=plot_signal_stats_flag)

            val_scores = {}
            for schemeset in self._configs.runtime.data_sampling_scheme_defs[VAL].keys():
                score = -self._run_epoch(epoch, VAL, schemeset, self._configs.runtime.loading.nbr_batches_val, save_imgs_flag=save_imgs_flag, plot_signals_flag=plot_signals_flag, plot_signal_stats_flag=plot_signal_stats_flag)
                if self._configs.runtime.data_sampling_scheme_defs[VAL][schemeset]['use_for_val_score']:
                    val_scores[schemeset] = score
            assert len(val_scores) > 0
            val_score = sum(val_scores.values()) / len(val_scores)

            self._lr_scheduler.step(val_score)
            # self._checkpoint_handler.save(self._model, epoch, train_score)
            self._checkpoint_handler.save(self._model, epoch, val_score)

    def eval(self):
        epoch = 1
        for schemeset, schemeset_def in self._configs.runtime.data_sampling_scheme_defs[TEST].items():
            schemeset_def = AttrDict(schemeset_def)
            visualization_config = self._configs.runtime.visualization
            if 'visualization' in schemeset_def:
                visualization_config += schemeset_def.visualization
            self._run_epoch(epoch, TEST, schemeset, self._configs.runtime.loading.nbr_batches_test, save_imgs_flag=visualization_config.save_imgs, plot_signals_flag=visualization_config.plot_signals, plot_signal_stats_flag=visualization_config.plot_signal_stats)

    def _sample_epoch_of_targets(self, mode, schemeset, nbr_batches):
        print('Running through epoch to collect target samples for prior...')
        selected_targets = {task_spec['target'] for task_name, task_spec in self._configs.tasks.items() if task_spec['prior_loss'] is not None}

        cnt = 1
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode, schemeset, nbr_batches * self._configs.runtime.loading.batch_size)):
            pertarget_target_features = self._loss_handler.get_target_features(batch.targets, selected_targets=selected_targets)
            # Map target features to corresponding tasks:
            target_features = self._loss_handler.map_features_to_tasks(pertarget_target_features)
            target_features_raw = self._loss_handler.apply_inverse_activation(target_features)
            self._loss_handler.record_batch_of_persample_signals('target_feat_raw', target_features_raw)
            if cnt % 10 == 0:
                print('{}/{}'.format(cnt, nbr_batches))
            cnt += 1

        target_samples = self._loss_handler.get_persample_signals_numpy()['target_feat_raw']
        self._loss_handler._reset_signals()
        print('Done.')
        return target_samples

    def _run_epoch(self, epoch, mode, schemeset, nbr_batches, save_imgs_flag=True, plot_signals_flag=True, plot_signal_stats_flag=True):
        if mode == TRAIN:
            self._model.train()
        else:
            self._model.eval()

        # cnt = 0
        visual_cnt = 1
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode, schemeset, nbr_batches * self._configs.runtime.loading.batch_size)):
            nn_out = self._run_model(batch.input, batch.extra_input)

            # Raw predicted features (neural net output)
            pred_features_raw = self._loss_handler.get_pred_features(nn_out)

            # Apply activation, and get corresponding target features
            pred_features = self._loss_handler.apply_activation(pred_features_raw)
            pertarget_target_features = self._loss_handler.get_target_features(batch.targets)
            # Map target features to corresponding tasks:
            target_features = self._loss_handler.map_features_to_tasks(pertarget_target_features)

            if self._configs.training.clamp_predictions:
                # Clamp features before loss computation (for the features where desired)
                pred_features = self._loss_handler.clamp_features(pred_features, before_loss=True)

            # Calculate loss
            task_loss_decays, loss_notapplied = self._loss_handler.calc_loss_decay(target_features, pertarget_target_features)
            if mode in (TRAIN, VAL):
                task_losses = self._loss_handler.calc_loss(pred_features, target_features, task_loss_decays)
                loss = sum(task_losses.values())
                if any([task_spec['prior_loss'] is not None for task_name, task_spec in self._configs.tasks.items()]):
                    prior_loss_signal_vals = self._loss_handler.calc_prior_loss(pred_features_raw, self._target_prior_samples)
                    loss += sum(prior_loss_signal_vals.values())

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
                    for sample_idx in range(self._configs.runtime.loading.batch_size):
                        self._visualizer.save_images(batch, pred_features, target_features, loss_notapplied, mode, schemeset, visual_cnt, sample=sample_idx)
                        visual_cnt += 1

        if save_imgs_flag:
            print('Saving images...')
            if mode in (TRAIN, VAL):
                self._visualizer.save_images(batch, pred_features, target_features, loss_notapplied, mode, schemeset, epoch, sample=-1)

        if plot_signals_flag:
            print('Plotting signals...')
            if mode in (TRAIN, VAL):
                self._visualizer.report_perbatch_signals(self._loss_handler.get_scalar_averages(), mode, schemeset, epoch)

        if plot_signal_stats_flag:
            print('Filtering samples where loss not applied...')
            self._loss_handler.filter_persample_signals([
                'interp_target_feat',
                'interp_pred_feat',
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

    def _run_model(self, inputs, extra_input):
        img1, img2 = inputs
        img1 = img1.to(get_device(), non_blocking=True)
        img2 = img2.to(get_device(), non_blocking=True)
        extra_input = extra_input.__class__(*tuple(map(lambda x: x.to(get_device(), non_blocking=True), extra_input)))
        return self._model((extra_input, img1, img2))


def run(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, args.train_or_eval)
    setup.prepare_environment()
    setup.save_settings(args)
    configs = setup.get_configs(args)

    main = Main(configs)
    # configs['data']['data_loader'] = main._data_loader
    if args.train_or_eval in 'train':
        main.train()
    else:
        main.eval()

if __name__ == '__main__':
    run(lib.setup)

import torch

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL
from lib.loss import LossHandler
from lib.model import Model
from lib.utils import get_device, get_configs, get_module_parameters
from lib.visualize import Visualizer
from lib.loader import Loader

class Trainer():
    """Trainer."""

    def __init__(self, configs):
        """Constructor."""
        self._configs = configs
        self._data_loader = Loader((TRAIN, VAL), self._configs)
        self._loss_handler = LossHandler(configs, self.__class__.__name__)
        self._checkpoint_handler = CheckpointHandler(configs)
        self._model = self._checkpoint_handler.init(self._init_model())
        self._optimizer = self._setup_optimizer()
        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max')
        self._visualizer = Visualizer(configs)

    def _init_model(self):
        model = Model(self._configs)
        return model

    def _setup_optimizer(self):
        weight_decay = self._configs.training.weight_decay if self._configs.training.weight_decay is not None else 0
        return torch.optim.Adam(
            self._model.parameters(),
            lr=self._configs.training.learning_rate,
            weight_decay=weight_decay,
        )

    def train(self):
        """Main loop."""
        for epoch in range(1, self._configs.training.n_epochs + 1):
            train_score = -self._run_epoch(epoch, TRAIN)
            val_score = -self._run_epoch(epoch, VAL)

            self._lr_scheduler.step(val_score)
            self._checkpoint_handler.save(self._model, epoch, train_score)
            # self._checkpoint_handler.save(self._model, epoch, val_score)

    def _run_epoch(self, epoch, mode):
        self._model.train()

        # cnt = 0
        # visual_cnt = 0
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode)):
            nn_out = self._run_model(batch.input, batch.extra_input)
            pred_features_raw = self._loss_handler.get_pred_features(nn_out)
            pred_features = self._loss_handler.apply_activation(pred_features_raw)
            target_features = self._loss_handler.get_target_features(batch.targets)
            if self._configs.training.clamp_predictions:
                # Clamp features before loss computation (for the features where desired)
                pred_features = self._loss_handler.clamp_features(pred_features, before_loss=True)
            task_loss_signal_vals = self._loss_handler.calc_loss(pred_features, target_features)
            loss = sum(task_loss_signal_vals.values())
            if self._configs.training.clamp_predictions:
                # Clamp features after loss computation (for all features)
                pred_features = self._loss_handler.clamp_features(pred_features, before_loss=False)
            interp_pred_features = self._loss_handler.calc_human_interpretable_features(pred_features)
            interp_target_features = self._loss_handler.calc_human_interpretable_features(target_features)
            interp_feat_abserror = self._loss_handler.calc_feature_abserrors(interp_pred_features, interp_target_features)
            interp_feat_abserror_avg = self._loss_handler.calc_batch_signal_avg(interp_feat_abserror)
            interp_feat_error = self._loss_handler.calc_feature_errors(interp_pred_features, interp_target_features)
            pred_feat_avg = self._loss_handler.calc_batch_signal_avg(pred_features)
            target_feat_avg = self._loss_handler.calc_batch_signal_avg(target_features)
            pred_feat_std = self._loss_handler.calc_batch_signal_std(pred_features)
            target_feat_std = self._loss_handler.calc_batch_signal_std(target_features)

            def flatten_and_stack(tensor_list):
                return torch.cat([x.reshape(-1) for x in tensor_list])

            w_params_all, b_params_all = get_module_parameters(self._model)
            w_params_final, b_params_final = self._model.get_last_layer_params()
            self._loss_handler.record_scalar_signals(
                'params/mean',
                {
                    'all_w_mean': flatten_and_stack(w_params_all).mean(),
                    'all_b_mean': flatten_and_stack(b_params_all).mean(),
                    'final_w_mean': flatten_and_stack(w_params_final).mean(),
                    'final_b_mean': flatten_and_stack(b_params_final).mean() if self._configs.model.head_layers[-1].bias else None,
                },
            )
            self._loss_handler.record_scalar_signals(
                'params/std',
                {
                    'all_w_std': flatten_and_stack(w_params_all).std(),
                    'all_b_std': flatten_and_stack(b_params_all).std(),
                    'final_w_std': flatten_and_stack(w_params_final).std(),
                    'final_b_std': flatten_and_stack(b_params_final).std() if self._configs.model.head_layers[-1].bias else None,
                },
            )

            self._loss_handler.record_scalar_signals('loss', {'loss': loss})
            self._loss_handler.record_scalar_signals('task_losses', task_loss_signal_vals)
            self._loss_handler.record_scalar_signals('interp_feat_abserror_avg', interp_feat_abserror_avg)
            self._loss_handler.record_scalar_signals('pred_feat_avg', pred_feat_avg)
            self._loss_handler.record_scalar_signals('target_feat_avg', target_feat_avg)
            self._loss_handler.record_scalar_signals('pred_feat_std', pred_feat_std)
            self._loss_handler.record_scalar_signals('target_feat_std', target_feat_std)

            self._loss_handler.record_tensor_signals('interp_feat_abserror', interp_feat_abserror)
            self._loss_handler.record_tensor_signals('interp_feat_error', interp_feat_error)
            self._loss_handler.record_tensor_signals('interp_pred_feat', interp_pred_features)
            self._loss_handler.record_tensor_signals('interp_target_feat', interp_target_features)
            self._loss_handler.record_tensor_signals('pred_feat', pred_features)
            self._loss_handler.record_tensor_signals('target_feat', target_features)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            # assert len(w_params_final) == 1 and len(b_params_final) == 1
            # w_params_final[0].data.clamp_(min=0.)
            # b_params_final[0].data.clamp_(min=0., max=0.)
            self._loss_handler.log_batch(epoch, batch_id, mode)

            # cnt += 1
            # if cnt % 10 == 0:
            #     self._visualizer.report_signals(self._loss_handler.get_scalar_averages(), mode)

            # if cnt % 30 == 0:
            #     visual_cnt += 1
            #     self._visualizer.save_images(batch, nn_out, mode, visual_cnt, sample=-1)
        self._visualizer.save_images(batch, pred_features, target_features, mode, epoch, sample=-1)

        self._visualizer.report_scalar_signals(self._loss_handler.get_scalar_averages(), mode, epoch)
        self._visualizer.calc_and_plot_signal_stats(self._loss_handler.get_signals_numpy(), mode, epoch)

        score = self._loss_handler.get_scalar_averages()['loss']['loss']
        self._loss_handler.finish_epoch(epoch, mode)
        return score

    def _run_model(self, inputs, extra_input):
        img1, img2 = inputs
        img1 = img1.to(get_device(), non_blocking=True)
        img2 = img2.to(get_device(), non_blocking=True)
        extra_input = extra_input.__class__(*tuple(map(lambda x: x.to(get_device(), non_blocking=True), extra_input)))
        return self._model((extra_input, img1, img2))


def main(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, 'train')
    setup.prepare_environment()
    setup.save_settings(args)

    configs = get_configs(args.config_name)
    configs += vars(args)
    trainer = Trainer(configs)
    # configs['data']['data_loader'] = trainer._data_loader
    trainer.train()

if __name__ == '__main__':
    main(lib.setup)

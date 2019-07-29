import torch

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL
from lib.loss import LossHandler
from lib.model import Model
from lib.utils import get_device, get_configs
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

    def _run_epoch(self, epoch, mode):
        self._model.train()

        # cnt = 0
        # visual_cnt = 0
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode)):
            nn_out = self._run_model(batch.input)
            pred_features, target_features = self._loss_handler.get_pred_and_target_features(nn_out, batch.targets)
            task_loss_signal_vals = self._loss_handler.calc_loss(pred_features, target_features)
            loss = sum(task_loss_signal_vals.values())
            interp_feat_error_signal_vals = self._loss_handler.calc_human_interpretable_feature_errors(pred_features, target_features)
            rel_feat_error_signal_vals = self._loss_handler.calc_relative_feature_errors(pred_features, target_features)
            self._loss_handler.record_signals('loss', {'loss': loss})
            self._loss_handler.record_signals('task_losses', task_loss_signal_vals)
            self._loss_handler.record_signals('interp_feat_error', interp_feat_error_signal_vals)
            self._loss_handler.record_signals('rel_feat_error', rel_feat_error_signal_vals)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._loss_handler.log_batch(epoch, batch_id, mode)

            # cnt += 1
            # if cnt % 10 == 0:
            #     self._visualizer.report_signals(self._loss_handler.get_averages(), mode)

            # if cnt % 30 == 0:
            #     visual_cnt += 1
            #     self._visualizer.save_images(batch, nn_out, mode, visual_cnt, sample=-1)
        self._visualizer.save_images(batch, pred_features, target_features, mode, epoch, sample=-1)

        self._visualizer.report_signals(self._loss_handler.get_averages(), mode)

        score = self._loss_handler.get_averages()['loss']['loss']
        self._loss_handler.finish_epoch(epoch, mode)
        return score

    def _run_model(self, inputs):
        img1, img2 = inputs
        img1 = img1.to(get_device(), non_blocking=True)
        img2 = img2.to(get_device(), non_blocking=True)
        return self._model((img1, img2))


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

"""Loss handler."""
import logging
from collections import defaultdict
import numpy as np
import torch
from torch import nn, exp, clamp

from lib.constants import TRAIN, VAL
from lib.utils import get_device, get_human_interp_maps


class LossHandler:
    """LossHandler."""
    def __init__(self, configs, name):
        self._configs = configs
        self._logger = logging.getLogger(name)
        self._signals = defaultdict(list)
        self._human_interp_maps = get_human_interp_maps(self._configs, 'torch')

        self._activation_dict = self._init_activations()
        self._loss_function_dict = self._init_loss_functions()

    def _init_activations(self):
        activation_dict = {}
        for task_name, task_spec in self._configs.tasks.items():
            if task_spec['activation'] == None:
                activation_dict[task_name] = None
            elif task_spec['activation'] == 'sigmoid':
                activation_dict[task_name] = nn.Sigmoid()
            else:
                raise NotImplementedError("{} loss not implemented.".format(task_spec['activation']))
        return activation_dict

    def _init_loss_functions(self):
        loss_function_dict = {}
        for task_name, task_spec in self._configs.tasks.items():
            if task_spec['loss_func'] == 'L1':
                # reduction='mean' averages over all dimensions (over the batch in this case)
                loss_function_dict[task_name] = nn.L1Loss(reduction='mean').to(get_device())
            elif task_spec['loss_func'] == 'L2':
                # reduction='mean' averages over all dimensions (over the batch in this case)
                loss_function_dict[task_name] = nn.MSELoss(reduction='mean').to(get_device())
            else:
                raise NotImplementedError("{} loss not implemented.".format(task_spec['loss_func']))
        return loss_function_dict

    def get_pred_and_target_features(self, nn_out, targets):
        pred_features = {}
        target_features = {}
        offset = 0
        for task_name in sorted(self._configs.tasks.keys()):
            pred_features[task_name] = nn_out[:, offset : offset + self._configs.tasks[task_name]['n_out']]
            if self._activation_dict[task_name] is not None:
                pred_features[task_name] = self._activation_dict[task_name](pred_features[task_name])
                if self._configs.tasks[task_name]['activation'] == 'sigmoid':
                    # Linearly map sigmoid output to desired range
                    assert self._configs.tasks[task_name]['min'] is not None and self._configs.tasks[task_name]['max'] is not None, \
                        'Min/max values mandatory when sigmoid activaiton is used'
                    pred_features[task_name] = pred_features[task_name] * (self._configs.tasks[task_name]['max'] - self._configs.tasks[task_name]['min'])
                    pred_features[task_name] = pred_features[task_name] + self._configs.tasks[task_name]['min']
            target_features[task_name] = getattr(targets, task_name).to(get_device())

            # Scalars may / may not introduce redundant dimension
            pred_features[task_name] = pred_features[task_name].squeeze()
            target_features[task_name] = target_features[task_name].squeeze()
            assert pred_features[task_name].shape == target_features[task_name].shape

            offset += self._configs.tasks[task_name]['n_out']

        return pred_features, target_features

    def calc_human_interpretable_feature_errors(self, pred_features, target_features):
        interp_feat_error_signal_vals = {}
        for task_name in sorted(self._configs.tasks.keys()):
            pred   = self._human_interp_maps[task_name](  pred_features[task_name])
            target = self._human_interp_maps[task_name](target_features[task_name])
            if len(pred.shape) == 1:
                interp_feat_error_signal_vals[task_name] = torch.mean(torch.abs(pred - target))
            else:
                assert len(pred.shape) == 2
                interp_feat_error_signal_vals[task_name] = torch.mean(torch.norm(pred - target, dim=1))
        return interp_feat_error_signal_vals

    def calc_loss(self, pred_features, target_features):
        # ======================================================================
        # TODO: Make sure CPU->GPU overhead is not too much.
        # Make sure "pin_memory" in dataloader works as intended.
        # Ideally custom batch & targets classes with pin_memory methods should be used, see: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        # ======================================================================

        task_loss_signal_vals = {}
        for task_name in sorted(self._configs.tasks.keys()):
            task_loss = self._loss_function_dict[task_name](
                pred_features[task_name],
                target_features[task_name],
            )
            task_loss = task_loss * self._configs.tasks[task_name]['loss_weight']
            task_loss_signal_vals[task_name] = task_loss
        return task_loss_signal_vals

    def record_signals(self, signal_vals, prefix=''):
        for signal_name, signal_val in signal_vals.items():
            self._signals[prefix + signal_name].append(signal_val)

    def get_averages(self, num_batches=0):
        avg_signals = defaultdict(int)
        for signal_name, signal_list in self._signals.items():
            latest_signals = signal_list[-num_batches:]
            avg_signals[signal_name] = (sum(latest_signals) / len(latest_signals)).detach().cpu().numpy()
        return avg_signals

    def log_batch(self, epoch, iteration, mode):
        """Log current batch."""
        losses = {
            'Loss': self.get_averages(num_batches=1),
            'Moving Avg': self.get_averages(num_batches=self._configs.logging.avg_window_size),
            'Average': self.get_averages(num_batches=0)
        }
        status_total_loss = ('[{name:s}]  '
                             'Epoch:{epoch:<3d}  '
                             'Iteration:{iteration:<5d}  '.
                             format(name=mode.upper(),
                                    epoch=epoch,
                                    iteration=iteration))
        # for statistic, value in losses.items():
        #     status_total_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
        #                                                              value=sum(value.values()))
        self._logger.info(status_total_loss)

        for task_name in self._signals.keys():
            status_task_loss = '{name:<45s}'.format(name=task_name)
            for statistic, value in losses.items():
                status_task_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                        value=value[task_name])
            self._logger.info(status_task_loss)

    def finish_epoch(self, epoch, mode):
        """Log current epoch."""
        mode = {TRAIN: 'Training', VAL: 'Validation'}[mode]
        self._logger.info('%s epoch %s done!',
                          mode, epoch)
        self._signals = defaultdict(list)

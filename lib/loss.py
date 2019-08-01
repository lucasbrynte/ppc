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
        self._tensor_signals = self._reset_signals()
        self._scalar_signals = self._reset_signals()
        self._human_interp_maps = get_human_interp_maps(self._configs, 'torch')

        self._activation_dict = self._init_activations()
        self._loss_function_dict = self._init_loss_functions()

    def _init_activations(self):
        activation_dict = {}
        for task_name, task_spec in self._configs.tasks.items():
            if task_spec['activation'] == None:
                activation_dict[task_name] = None
            elif task_spec['activation'] == 'square':
                activation_dict[task_name] = lambda x: x**2
            elif task_spec['activation'] == 'abs':
                activation_dict[task_name] = lambda x: torch.abs(x)
            elif task_spec['activation'] == 'sigmoid':
                activation_dict[task_name] = nn.Sigmoid()
            else:
                raise NotImplementedError("{} loss not implemented.".format(task_spec['activation']))
        return activation_dict

    def _reset_signals(self):
        return defaultdict(lambda: defaultdict(list))

    def _init_loss_functions(self):
        loss_function_dict = {}
        for task_name, task_spec in self._configs.tasks.items():
            if task_spec['loss_func'] == 'L1':
                loss_function_dict[task_name] = nn.L1Loss(reduction='none').to(get_device())
            elif task_spec['loss_func'] == 'L2':
                loss_function_dict[task_name] = nn.MSELoss(reduction='none').to(get_device())
            elif task_spec['loss_func'] == 'BCE':
                loss_function_dict[task_name] = nn.BCELoss(reduction='none').to(get_device())
            else:
                raise NotImplementedError("{} loss not implemented.".format(task_spec['loss_func']))
        return loss_function_dict

    def get_pred_features(self, nn_out):
        # Batch size determined and preserved to be used in other methods. May vary when switching between TRAIN / VAL
        self._batch_size = nn_out.shape[0]

        pred_features = {}
        offset = 0
        for task_name in sorted(self._configs.tasks.keys()):
            pred_features[task_name] = nn_out[:, offset : offset + self._configs.targets[self._configs.tasks[task_name]['target']]['n_out']]
            offset += self._configs.targets[self._configs.tasks[task_name]['target']]['n_out']

        return pred_features

    def get_target_features(self, targets, selected_tasks=None):
        if selected_tasks is None:
            selected_tasks = self._configs.tasks.keys()
        target_features = {}
        for task_name in sorted(selected_tasks):
            target_features[task_name] = getattr(targets, self._configs.tasks[task_name]['target']).to(get_device())
        return target_features

    def apply_activation(self, pred_features_raw):
        pred_features = pred_features_raw.copy() # Shallow copy
        for task_name in pred_features.keys():
            if self._activation_dict[task_name] is not None:
                pred_features[task_name] = self._activation_dict[task_name](pred_features_raw[task_name])
                if self._configs.tasks[task_name]['activation'] == 'sigmoid':
                    # Linearly map sigmoid output to desired range
                    assert self._configs.targets[self._configs.tasks[task_name]['target']]['min'] is not None and self._configs.targets[self._configs.tasks[task_name]['target']]['max'] is not None, \
                        'Min/max values mandatory when sigmoid activaiton is used'
                    pred_features[task_name] = pred_features[task_name] * (self._configs.targets[self._configs.tasks[task_name]['target']]['max'] - self._configs.targets[self._configs.tasks[task_name]['target']]['min'])
                    pred_features[task_name] = pred_features[task_name] + self._configs.targets[self._configs.tasks[task_name]['target']]['min']
        return pred_features

    def apply_inverse_activation(self, pred_features):
        pred_features_raw = pred_features.copy() # Shallow copy
        for task_name in pred_features_raw.keys():
            if self._activation_dict[task_name] is not None:
                if self._configs.tasks[task_name]['activation'] == 'square':
                    sqrt_feat = torch.sqrt(pred_features[task_name])
                    pred_features_raw[task_name] = torch.cat([sqrt_feat, -sqrt_feat], dim=0)
                elif self._configs.tasks[task_name]['activation'] == 'abs':
                    feat = pred_features[task_name]
                    pred_features_raw[task_name] = torch.cat([feat, -feat], dim=0)
                else:
                    assert False
        return pred_features_raw

    def clamp_features(self, features, before_loss=False):
        clamped_features = features.copy() # Shallow copy
        for task_name in self._configs.tasks.keys():
            if before_loss and not self._configs.tasks[task_name]['clamp_before_loss']:
                continue
            if self._configs.targets[self._configs.tasks[task_name]['target']]['min'] is None and self._configs.targets[self._configs.tasks[task_name]['target']]['max'] is None:
                continue
            clamped_features[task_name] = torch.clamp(features[task_name], min=self._configs.targets[self._configs.tasks[task_name]['target']]['min'], max=self._configs.targets[self._configs.tasks[task_name]['target']]['max'])
        return clamped_features

    def calc_human_interpretable_features(self, features):
        interp_features = {}
        for task_name in self._configs.tasks.keys():
            interp_features[task_name] = self._human_interp_maps[task_name](features[task_name])
        return interp_features

    def calc_feature_abserrors(self, pred_features, target_features):
        feat_error_signal_vals = {}
        for task_name in self._configs.tasks.keys():
            if len(pred_features[task_name].shape) == 1:
                feat_error_signal_vals[task_name] = torch.abs(pred_features[task_name] - target_features[task_name])
            else:
                assert len(pred_features[task_name].shape) == 2
                feat_error_signal_vals[task_name] = torch.norm(pred_features[task_name] - target_features[task_name], dim=1)
        return feat_error_signal_vals

    def calc_feature_errors(self, pred_features, target_features):
        feat_resid_signal_vals = {}
        for task_name in self._configs.tasks.keys():
            feat_resid_signal_vals[task_name] = pred_features[task_name] - target_features[task_name]
        return feat_resid_signal_vals

    def calc_batch_signal_avg(self, signals):
        feat_avg_signal_vals = {}
        for task_name in signals:
            feat_avg_signal_vals[task_name] = torch.mean(signals[task_name])
        return feat_avg_signal_vals

    def calc_batch_signal_std(self, signals):
        feat_std_signal_vals = {}
        for task_name in signals:
            feat_std_signal_vals[task_name] = torch.std(signals[task_name])
        return feat_std_signal_vals

    def calc_loss(self, pred_features, target_features):
        # ======================================================================
        # TODO: Make sure CPU->GPU overhead is not too much.
        # Make sure "pin_memory" in dataloader works as intended.
        # Ideally custom batch & targets classes with pin_memory methods should be used, see: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        # ======================================================================

        task_loss_signal_vals = {}
        for task_name in self._configs.tasks.keys():
            assert pred_features[task_name].shape == target_features[task_name].shape
            if self._configs.tasks[task_name]['loss_func'] == 'BCE' and self._configs.tasks[task_name]['activation'] == 'sigmoid':
                # Linearly map output back to [0, 1] range
                assert self._configs.targets[self._configs.tasks[task_name]['target']]['min'] is not None and self._configs.targets[self._configs.tasks[task_name]['target']]['max'] is not None, \
                    'Min/max values mandatory when sigmoid activaiton is used'
                task_loss = self._loss_function_dict[task_name](
                    (pred_features[task_name] - self._configs.targets[self._configs.tasks[task_name]['target']]['min']) / (self._configs.targets[self._configs.tasks[task_name]['target']]['max'] - self._configs.targets[self._configs.tasks[task_name]['target']]['min']),
                    (target_features[task_name] - self._configs.targets[self._configs.tasks[task_name]['target']]['min']) / (self._configs.targets[self._configs.tasks[task_name]['target']]['max'] - self._configs.targets[self._configs.tasks[task_name]['target']]['min']),
                )
            else:
                task_loss = self._loss_function_dict[task_name](
                    pred_features[task_name],
                    target_features[task_name],
                )
            if self._configs.tasks[task_name]['target_norm_loss_decay'] is None:
                pass
            elif self._configs.tasks[task_name]['target_norm_loss_decay']['method'] == 'relative':
                assert len(target_features[task_name].shape) == 2
                task_loss = task_loss / (torch.norm(target_features[task_name], dim=1, keepdim=True) + self._configs.tasks[task_name]['target_norm_loss_decay']['min_denominator'])
            elif self._configs.tasks[task_name]['target_norm_loss_decay']['method'] == 'exp_decay':
                assert len(target_features[task_name].shape) == 2
                gamma = np.log(2.0) / self._configs.tasks[task_name]['target_norm_loss_decay']['halflife']
                task_loss = task_loss * torch.exp(-gamma * torch.norm(target_features[task_name], dim=1, keepdim=True))
            task_loss = task_loss * self._configs.tasks[task_name]['loss_weight']
            task_loss = task_loss.mean() # So far loss is element-wise. Reduce over entire batch.
            task_loss_signal_vals[task_name] = task_loss
        return task_loss_signal_vals

    def record_tensor_signals(self, tag, signal_vals):
        """
        Takes a dict of GPU pytorch tensors, with 1st dimension over samples in batch.
        Splits the tensors into list of samples, and concatenates this list onto their corresponding signal lists
        """
        assert tag not in self._scalar_signals.keys()
        for signal_name, signal_val in signal_vals.items():
            self._tensor_signals[tag][signal_name] += list(signal_val)

    def record_scalar_signals(self, tag, signal_vals):
        """
        Takes a dict of GPU pytorch scalars, and appends them to their corresponding signal lists
        """
        assert tag not in self._tensor_signals.keys()
        for signal_name, signal_val in signal_vals.items():
            self._scalar_signals[tag][signal_name].append(signal_val)

    def get_signals_numpy(self, tag_filter=None):
        numpy_signals = defaultdict(lambda: defaultdict(float))
        for signals_list in [self._tensor_signals, self._scalar_signals]:
            for tag in signals_list:
                if tag_filter is not None and tag != tag_filter:
                    continue
                for signal_name, signal_list in signals_list[tag].items():
                    numpy_signals[tag][signal_name] = torch.tensor(signal_list).detach().cpu().numpy()
        return numpy_signals

    def get_scalar_averages(self, num_batches=0, tag_filter=None):
        avg_signals = defaultdict(lambda: defaultdict(float))
        for tag in self._scalar_signals:
            if tag_filter is not None and tag != tag_filter:
                continue
            for signal_name, signal_list in self._scalar_signals[tag].items():
                latest_signals = signal_list[-num_batches:]
                avg_signals[tag][signal_name] = (sum(latest_signals) / len(latest_signals)).detach().cpu().numpy()
        return avg_signals

    def log_batch(self, epoch, iteration, mode):
        """Log current batch."""
        losses = {
            'Raw': self.get_scalar_averages(num_batches=1),
            'Moving Avg': self.get_scalar_averages(num_batches=self._configs.logging.avg_window_size),
            'Average': self.get_scalar_averages(num_batches=0)
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

        for tag in self._scalar_signals:
            for signal_name, signal_list in self._scalar_signals[tag].items():
                status_task_loss = '{name:<45s}'.format(name=tag+'/'+signal_name)
                for statistic, value in losses.items():
                    status_task_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                        value=value[tag][signal_name])
                self._logger.info(status_task_loss)

    def finish_epoch(self, epoch, mode):
        """Log current epoch."""
        mode = {TRAIN: 'Training', VAL: 'Validation'}[mode]
        self._logger.info('%s epoch %s done!',
                          mode, epoch)
        self._tensor_signals = self._reset_signals()
        self._scalar_signals = self._reset_signals()

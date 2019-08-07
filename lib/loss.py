"""Loss handler."""
import logging
from collections import defaultdict
import numpy as np
import torch
from torch import nn, exp, clamp
import geomloss

from lib.constants import TRAIN, VAL
from lib.utils import get_device, get_human_interp_maps


class LossHandler:
    """LossHandler."""
    def __init__(self, configs, name):
        self._configs = configs
        self._logger = logging.getLogger(name)

        self._reset_signals()
        self._human_interp_maps = get_human_interp_maps(self._configs, 'torch')
        self._activation_dict = self._init_activations()
        self._loss_function_dict = self._init_loss_functions()
        if any([task_spec['prior_loss'] is not None for task_name, task_spec in self._configs.tasks.items()]):
            self._sinkhorn_loss = geomloss.SamplesLoss(
                loss = 'sinkhorn',
                p = 2,
                blur = 0.05,
            )

    def _init_activations(self):
        activation_dict = {}
        for target_name, target_spec in self._configs.targets.items():
            if target_spec['activation'] == None:
                activation_dict[target_name] = None
            elif target_spec['activation'] == 'square':
                activation_dict[target_name] = lambda x: x**2
            elif target_spec['activation'] == 'abs':
                activation_dict[target_name] = lambda x: torch.abs(x)
            elif target_spec['activation'] == 'sigmoid':
                activation_dict[target_name] = nn.Sigmoid()
            else:
                raise NotImplementedError("{} loss not implemented.".format(target_spec['activation']))
        return activation_dict

    def _get_signals_defaultdict(self):
        return defaultdict(lambda: defaultdict(list))

    def _reset_signals(self):
        self._persample_signals = self._get_signals_defaultdict()
        self._perbatch_signals = self._get_signals_defaultdict()

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

    def get_target_features(self, targets, selected_targets=None):
        if selected_targets is None:
            selected_targets = self._configs.targets.keys()
        target_features = {}
        for target_name in sorted(selected_targets):
            target_features[target_name] = getattr(targets, target_name).to(get_device())
        return target_features

    def map_features_to_tasks(self, features):
        pertask_features = {}
        for task_name in self._configs.tasks.keys():
            target_name = self._configs.tasks[task_name]['target']
            pertask_features[task_name] = features[target_name]
        return pertask_features

    def apply_activation(self, pred_features_raw):
        pred_features = pred_features_raw.copy() # Shallow copy
        for task_name in pred_features.keys():
            target_name = self._configs.tasks[task_name]['target']
            if self._activation_dict[target_name] is not None:
                pred_features[task_name] = self._activation_dict[target_name](pred_features_raw[task_name])
                if self._configs.targets[target_name]['activation'] == 'sigmoid':
                    # Linearly map sigmoid output to desired range
                    assert self._configs.targets[target_name]['min'] is not None and self._configs.targets[target_name]['max'] is not None, \
                        'Min/max values mandatory when sigmoid activaiton is used'
                    pred_features[task_name] = pred_features[task_name] * (self._configs.targets[target_name]['max'] - self._configs.targets[target_name]['min'])
                    pred_features[task_name] = pred_features[task_name] + self._configs.targets[target_name]['min']
        return pred_features

    def apply_inverse_activation(self, pred_features):
        pred_features_raw = pred_features.copy() # Shallow copy
        for task_name in pred_features_raw.keys():
            target_name = self._configs.tasks[task_name]['target']
            if self._activation_dict[target_name] is not None:
                if self._configs.targets[target_name]['activation'] == 'square':
                    sqrt_feat = torch.sqrt(pred_features[task_name])
                    pred_features_raw[task_name] = torch.cat([sqrt_feat, -sqrt_feat], dim=0)
                elif self._configs.targets[target_name]['activation'] == 'abs':
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

    def calc_norm(self, to_be_normed):
        feat_error_signal_vals = {}
        for task_name in self._configs.tasks.keys():
            if len(to_be_normed[task_name].shape) == 1:
                feat_error_signal_vals[task_name] = torch.abs(to_be_normed[task_name])
            else:
                assert len(to_be_normed[task_name].shape) == 2
                feat_error_signal_vals[task_name] = torch.norm(to_be_normed[task_name], dim=1)
        return feat_error_signal_vals

    def calc_feature_errors(self, pred_features, target_features):
        feat_resid_signal_vals = {}
        for task_name in self._configs.tasks.keys():
            feat_resid_signal_vals[task_name] = pred_features[task_name] - target_features[task_name]
        return feat_resid_signal_vals

    def normalize_interp_vals(self, interp_to_be_normalized, interp_feat_norms):
        feat_error_signal_vals = {}
        for task_name in self._configs.tasks.keys():
            assert interp_to_be_normalized[task_name].shape == interp_feat_norms[task_name].shape

            if self._configs.targets[self._configs.tasks[task_name]['target']]['unit'] == 'log_factor':
                # Not sure how to handle these features here...
                continue

            # Division by norm, unless too close to zero (denominator is saturated from below).
            MIN_DENOM_VAL = {
                # All values are interpretable, i.e. degrees etc.
                'angle': 0.1,
                'cosdist': 0.1,
                'px': 0.1,
                # 'log_factor': 0.1,
            }[self._configs.targets[self._configs.tasks[task_name]['target']]['unit']]

            if len(interp_to_be_normalized[task_name].shape) == 1:
                feat_error_signal_vals[task_name] = interp_to_be_normalized[task_name] / torch.abs(interp_feat_norms[task_name]).clamp(MIN_DENOM_VAL)
            else:
                assert len(interp_to_be_normalized[task_name].shape) == 2
                feat_error_signal_vals[task_name] = interp_to_be_normalized[task_name] / torch.norm(interp_feat_norms[task_name], dim=1).clamp(MIN_DENOM_VAL)
        return feat_error_signal_vals

    def calc_batch_signal_avg(self, signals, discard_signal=None):
        feat_avg_signal_vals = {}
        for task_name in signals:
            if discard_signal is None:
                feat_avg_signal_vals[task_name] = torch.mean(signals[task_name])
            else:
                assert discard_signal[task_name].squeeze(1).shape == signals[task_name].shape
                feat_avg_signal_vals[task_name] = torch.mean(signals[task_name][~discard_signal[task_name].squeeze(1)])
        return feat_avg_signal_vals

    def calc_batch_signal_std(self, signals, discard_signal=None):
        feat_std_signal_vals = {}
        for task_name in signals:
            if discard_signal is None:
                feat_std_signal_vals[task_name] = torch.std(signals[task_name])
            else:
                assert discard_signal[task_name].squeeze(1).shape == signals[task_name].shape
                feat_std_signal_vals[task_name] = torch.std(signals[task_name][~discard_signal[task_name].squeeze(1)])
        return feat_std_signal_vals

    def calc_decay_factor(self, decay_spec, decay_controlling_variable):
        if decay_spec is None:
            return 1.

        assert len(decay_controlling_variable.shape) == 2
        decay_controlling_variable = torch.norm(decay_controlling_variable, dim=1, keepdim=True)

        if decay_spec['method'] == 'relative':
            return 1. / (decay_controlling_variable + decay_spec['min_denominator'])

        if decay_spec['method'] == 'exp_decay':
            gamma = np.log(2.0) / decay_spec['halflife']
            return torch.exp(-gamma * decay_controlling_variable)

        if decay_spec['method'] == 'smoothstep':
            x1, x2, y1, y2 = decay_spec['x1'], decay_spec['x2'], decay_spec['y1'], decay_spec['y2']
            assert x2 > x1
            x = decay_controlling_variable.clamp(x1, x2)
            return y1 + (y2-y1) * 0.5 * (1.0 - torch.cos((x-x1) * np.pi/(x2-x1)))

        assert False

    def calc_loss_decay(self, target_features, pertarget_target_features):
        task_loss_decays = {}
        loss_notapplied = {}
        for task_name in self._configs.tasks.keys():

            # Initialize decay to 1.0:
            loss_decay = 1.0

            # Assume loss applied for every sample until proven wrong:
            loss_notapplied_mask = torch.zeros_like(target_features[task_name], dtype=torch.uint8)

            if self._configs.tasks[task_name]['loss_decay'] is not None:
                for decay_spec in self._configs.tasks[task_name]['loss_decay']:
                    decay_controlling_variable = target_features[task_name] if not 'target' in decay_spec else pertarget_target_features[decay_spec['target']]
                    loss_decay *= self.calc_decay_factor(decay_spec, decay_controlling_variable)

                    if decay_controlling_variable is not target_features[task_name]:
                        # A target other than itself is being used to control loss decay
                        if decay_spec['method'] == 'smoothstep':
                            assert decay_spec['y2'] < decay_spec['y1']
                            loss_notapplied_mask |= (decay_controlling_variable > 0.5*(decay_spec['x1']+decay_spec['x2']))

            task_loss_decays[task_name] = loss_decay
            loss_notapplied[task_name] = loss_notapplied_mask

        return task_loss_decays, loss_notapplied

    def calc_loss(self, pred_features, target_features, task_loss_decays):
        # ======================================================================
        # TODO: Make sure CPU->GPU overhead is not too much.
        # Make sure "pin_memory" in dataloader works as intended.
        # Ideally custom batch & targets classes with pin_memory methods should be used, see: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        # ======================================================================

        task_losses = {}
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
            task_loss = task_loss * task_loss_decays[task_name]
            task_loss = task_loss * self._configs.tasks[task_name]['loss_weight']
            task_loss = task_loss.mean() # So far loss is element-wise. Reduce over entire batch.
            task_losses[task_name] = task_loss
        return task_losses

    def calc_prior_loss(self, pred_features_raw, target_prior_samples):
        prior_loss_signal_vals = {}
        for task_name, task_spec in self._configs.tasks.items():
            if task_spec['prior_loss'] is not None:
                assert task_spec['prior_loss']['method'] == 'sinkhorn'
                loss_weight = task_spec['loss_weight'] * task_spec['prior_loss']['loss_weight']
                prior_loss_signal_vals[task_name] = loss_weight * self._sinkhorn_loss(pred_features_raw[task_name], target_prior_samples[task_name].reshape(-1,1))
        return prior_loss_signal_vals

    def record_batch_of_persample_signals(self, signal_name, signal_vals):
        """
        Takes a dict of GPU pytorch tensors, with 1st dimension over samples in batch.
        Splits the tensors into list of samples, and concatenates this list onto their corresponding signal lists
        """
        assert signal_name not in self._perbatch_signals.keys()
        for task_name, signal_val in signal_vals.items():
            self._persample_signals[signal_name][task_name] += list(signal_val)

    def record_batch_of_perbatch_signals(self, signal_name, signal_vals):
        """
        Takes a dict of GPU pytorch scalars, and appends them to their corresponding signal lists
        """
        assert signal_name not in self._persample_signals.keys()
        for task_name, signal_val in signal_vals.items():
            self._perbatch_signals[signal_name][task_name].append(signal_val)

    def filter_tensor_signal(self, signal_vals, loss_notapplied):
        signal_vals_filtered = {}
        for task_name in signal_vals.keys():
            signal_vals_filtered[task_name] = [sample_val for sample_val, discard_val in zip(signal_vals[task_name], loss_notapplied[task_name]) if not torch.any(discard_val)]
        return signal_vals_filtered

    def filter_persample_signals(self, signal_names):
        suffix = '_filtered'
        filtered_persample_signals = {}
        for signal_name in signal_names:
            filtered_persample_signals[signal_name + suffix] = self.filter_tensor_signal(self._persample_signals[signal_name], self._persample_signals['loss_notapplied'])
        self._persample_signals.update(filtered_persample_signals)

    def get_perbatch_signals(self):
        return self._perbatch_signals

    def get_persample_signals(self):
        return self._persample_signals

    def get_signals_numpy(self, signal_dict, signal_name_filter=None):
        numpy_signals = defaultdict(lambda: defaultdict(float))
        for signal_name in signal_dict:
            if signal_name_filter is not None and signal_name != signal_name_filter:
                continue
            for task_name, samples_list in signal_dict[signal_name].items():
                numpy_signals[signal_name][task_name] = torch.tensor(samples_list).detach().cpu().numpy()
        return numpy_signals

    def get_perbatch_signals_numpy(self, signal_name_filter=None):
        return self.get_signals_numpy(self._perbatch_signals, signal_name_filter=signal_name_filter)

    def get_persample_signals_numpy(self, signal_name_filter=None):
        return self.get_signals_numpy(self._persample_signals, signal_name_filter=signal_name_filter)

    def get_scalar_averages(self, num_batches=0, signal_name_filter=None):
        avg_signals = defaultdict(lambda: defaultdict(float))
        for signal_name in self._perbatch_signals:
            if signal_name_filter is not None and signal_name != signal_name_filter:
                continue
            for task_name, samples_list in self._perbatch_signals[signal_name].items():
                latest_samples = samples_list[-num_batches:]
                avg_signals[signal_name][task_name] = (sum(latest_samples) / len(latest_samples)).detach().cpu().numpy()
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

        for signal_name in self._perbatch_signals:
            for task_name, samples_list in self._perbatch_signals[signal_name].items():
                status_task_loss = '{name:<45s}'.format(name=signal_name+'/'+task_name)
                for statistic, value in losses.items():
                    status_task_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                        value=value[signal_name][task_name])
                self._logger.info(status_task_loss)

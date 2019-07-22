"""Loss handler."""
import logging
from collections import defaultdict
import torch
from torch import nn, exp, clamp

from lib.constants import TRAIN, VAL
from lib.utils import get_device


class LossHandler:
    """LossHandler."""
    def __init__(self, configs, name):
        self._configs = configs
        self._logger = logging.getLogger(name)
        self._losses = defaultdict(list)

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

    def calc_loss(self, nn_out, annotation):
        # nn_out shape: (batch_size, out_features)
        # annotation shapes: (batch_size, <ANNO_SHAPE>)

        loss = 0

        offset = 0
        for task_name, task_spec in self._configs.tasks.items():
            pred_features = nn_out[:, offset : offset + task_spec['n_out']]
            if self._activation_dict[task_name] is not None:
                pred_features = self._activation_dict[task_name](pred_features)
            target_features = getattr(annotation, task_name).to(get_device())

            # Scalars may / may not introduce redundant dimension
            pred_features = pred_features.squeeze()
            target_features = target_features.squeeze()
            assert pred_features.shape == target_features.shape

            task_loss = self._loss_function_dict[task_name](
                pred_features,
                target_features,
            )
            self._losses[task_name].append(task_loss)
            loss += task_loss
            offset += task_spec['n_out']

        return loss

    def get_averages(self, num_batches=0):
        avg_losses = defaultdict(int)
        for loss_name, loss_list in self._losses.items():
            latest_losses = loss_list[-num_batches:]
            avg_losses[loss_name] = sum(latest_losses) / len(latest_losses)
        return avg_losses

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
        for statistic, value in losses.items():
            status_total_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                     value=sum(value.values()))
        self._logger.info(status_total_loss)

        for task_name in self._losses.keys():
            status_task_loss = '{name:<26s}'.format(name=task_name)
            for statistic, value in losses.items():
                status_task_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                        value=value[task_name])
            self._logger.info(status_task_loss)

    def finish_epoch(self, epoch, mode):
        """Log current epoch."""
        mode = {TRAIN: 'Training', VAL: 'Validation'}[mode]
        self._logger.info('%s epoch %s done!',
                          mode, epoch)
        self._losses = defaultdict(list)

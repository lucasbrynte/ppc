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
        self._l1_loss = nn.L1Loss(reduction='mean').to(get_device()) # reduction='mean' averages over all dimensions (over the batch in this case)

    def calc_loss(self, nn_out, annotation):
        # Shape: (batch_size, 1)
        gt = torch.norm(annotation.delta_loc, dim=1, keepdim=True).to(get_device())
        loss = self._l1_loss(nn_out, gt)
        self._losses['loss'].append(loss)
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

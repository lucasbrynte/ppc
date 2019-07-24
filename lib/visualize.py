import os
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import math
import numpy as np

import torch
from torchvision.transforms.functional import normalize
from tensorboardX import SummaryWriter

from lib.constants import PYPLOT_DPI
from lib.constants import TV_MEAN, TV_STD


class Visualizer:
    """Visualizer."""
    def __init__(self, configs):
        self._configs = configs
        vis_path = os.path.join(configs.experiment_path, 'visual')
        shutil.rmtree(vis_path, ignore_errors=True)
        self._writer = SummaryWriter(vis_path)
        self._loss_count_dict = {'train': 0, 'val': 0}

    def __del__(self):
        # Unsure of the importance of calling close()... Might not be done in case of KeyboardInterrupt
        # https://stackoverflow.com/questions/44831317/tensorboard-unble-to-get-first-event-timestamp-for-run
        # https://stackoverflow.com/questions/33364340/how-to-avoid-suppressing-keyboardinterrupt-during-garbage-collection-in-python
        self._writer.close()

    def report_loss(self, losses, mode):
        self._writer.add_scalar('loss/{}'.format(mode), sum(losses.values()), self._loss_count_dict[mode])
        self._writer.add_scalars('task_losses/{}'.format(mode), losses, self._loss_count_dict[mode])
        self._loss_count_dict[mode] += 1

    def _retrieve_input_img(self, image_tensor):
        img = normalize(image_tensor, mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        img = torch.clamp(img, 0.0, 1.0)
        img = np.moveaxis(img.numpy(), 0, -1)
        return img

    def _plot_img(self, ax, img, title, bbox2d=None):
        img = np.clip(img, 0.0, 1.0)
        if bbox2d is None:
            ax.axis('on')
            ax.set_xlim(-0.5,                                  -0.5 + self._configs.data.crop_dims[1])
            ax.set_ylim(-0.5 + self._configs.data.crop_dims[0], -0.5)
        else:
            x1, y1, x2, y2 = bbox2d
            ax.set_xlim(x1, x2)
            ax.set_ylim(y2, y1)
        ax.autoscale(enable=False)
        ax.imshow(img)
        ax.set_title(title)

    def _plot_text(self, ax, text, fontsize=10):
        ax.axis('off')
        ax.axis([0, 10, 0, 10])
        ax.text(0, 10, text, verticalalignment='top', horizontalalignment='left', wrap=True, fontsize=fontsize, fontfamily='monospace')

    def _pretty_print_feature_value(self, task_name, feat):
        feat = np.array(feat)
        ndims = len(feat.shape)
        if ndims == 0:
            feat = feat[None] # Unsqueeze
        else:
            assert ndims == 1, "Pretty printing with feature dimensions {} unsupported.".format(feat.shape)

        unit = self._configs.tasks[task_name]['unit']
        if unit == 'px':
            print_elem = lambda x: '{:.2f}'.format(x)
            unit_suffix = ' px'
        elif unit == 'angle':
            print_elem = lambda x: '{:.1f}'.format(x * 180./math.pi)
            unit_suffix = ' deg'
        elif unit == 'cosdist':
            print_elem = lambda x: '{:.1f}'.format(np.arccos(1.0-x) * 180./math.pi)
            unit_suffix = ' deg'
        elif unit == 'log_factor':
            print_elem = lambda x: '{:.2f}'.format(np.exp(x))
            unit_suffix = ' factor'
        else:
            return '{}'.format(feat)

        tmp = ', '.join([print_elem(val) for val in feat])
        if len(feat) > 1:
            tmp = '(' + tmp + ')'
        return tmp + unit_suffix

    def save_images(self, batch, pred_features, target_features, mode, step_index, sample=-1):
        img1_batch, img2_batch = batch.input
        img_shape = img1_batch.shape[-2:]
        img1 = self._retrieve_input_img(img1_batch[sample])
        img2 = self._retrieve_input_img(img2_batch[sample])

        fig, axes_array = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=[10, 10],
            # figsize=[img_shape[1] / PYPLOT_DPI, img_shape[0] / PYPLOT_DPI],
            squeeze=False,
            dpi=PYPLOT_DPI,
            tight_layout=True,
            gridspec_kw={'height_ratios': [1, 2]},
        )
        self._plot_img(axes_array[0,0], img1, 'Ref. image')
        self._plot_img(axes_array[0,1], img2, 'Query image')

        def task_lines(task_name):
            lines = [
                task_name,
                '{:<8s}{:<8s} {:s}'.format('-', 'pred:', self._pretty_print_feature_value(task_name, pred_features[task_name][sample].detach().cpu().numpy())),
                '{:<8s}{:<8s} {:s}'.format('-', 'target:', self._pretty_print_feature_value(task_name, target_features[task_name][sample].detach().cpu().numpy())),
            ]
            return lines
        lines = [line for task_name in sorted(self._configs.tasks.keys()) for line in task_lines(task_name)]
        text = '\n'.join(lines)

        gs = axes_array[1,0].get_gridspec() # Might as well have taken gridspec from any axes..?
        axes_array[1,0].remove()
        axes_array[1,1].remove()
        row1_ax = fig.add_subplot(gs[1, :])
        self._plot_text(row1_ax, text)

        self._writer.add_figure(mode, fig, step_index)

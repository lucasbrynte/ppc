import os
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
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
        ax.text(0, 10, text, verticalalignment='top', horizontalalignment='left', wrap=True, fontsize=fontsize)

    def save_images(self, batch, nn_out, mode, step_index, sample=-1):
        img1_batch, img2_batch = batch.input
        img_shape = img1_batch.shape[-2:]
        img1 = self._retrieve_input_img(img1_batch[sample])
        img2 = self._retrieve_input_img(img2_batch[sample])

        fig, axes_array = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=[8, 8],
            # figsize=[img_shape[1] / PYPLOT_DPI, img_shape[0] / PYPLOT_DPI],
            squeeze=False,
            dpi=PYPLOT_DPI,
            tight_layout=True,
        )
        self._plot_img(axes_array[0,0], img1, 'Ref. image')
        self._plot_img(axes_array[1,0], img2, 'Query image')
        self._plot_text(axes_array[0,1], ' '.join(100*['Hello1']))
        self._plot_text(axes_array[1,1], ' '.join(100*['Hello2']))
        self._writer.add_figure(mode, fig, step_index)

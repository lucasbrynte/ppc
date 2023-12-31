import os
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import math
import numpy as np

import torch
from torchvision.transforms.functional import normalize
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from lib.utils import get_human_interp_maps
from lib.constants import PYPLOT_DPI
from lib.constants import TV_MEAN, TV_STD


class Visualizer:
    """Visualizer."""
    def __init__(self, configs):
        self._configs = configs
        vis_path = os.path.join(configs.experiment_path, 'visual')
        shutil.rmtree(vis_path, ignore_errors=True)
        self._writer = SummaryWriter(vis_path)
        self._human_interp_maps = get_human_interp_maps(self._configs, 'numpy')

    def __del__(self):
        # Unsure of the importance of calling close()... Might not be done in case of KeyboardInterrupt
        # https://stackoverflow.com/questions/44831317/tensorboard-unble-to-get-first-event-timestamp-for-run
        # https://stackoverflow.com/questions/33364340/how-to-avoid-suppressing-keyboardinterrupt-during-garbage-collection-in-python
        self._writer.close()

    def report_perbatch_signals(self, avg_signals, mode, schemeset, step_index):
        for tag in avg_signals:
            self._writer.add_scalars('{}/{}_{}'.format(tag, mode, schemeset), avg_signals[tag], step_index)

    def _group_feature_error_data_into_bins_wrt_target_magnitude(self, interp_target_feat, interp_feat_error):
        """
        Groups the "interp_feat_error" signals, based on the magnitude of the corresponding target values
        """
        binned_signals = defaultdict(lambda: [None]*nbr_bins)
        bin_edges = {}

        assert set(interp_target_feat.keys()) >= set(interp_feat_error.keys())
        for task_name in sorted(interp_feat_error.keys()):
            nbr_samples = interp_target_feat[task_name].shape[0]
            target_magnitudes = np.linalg.norm(interp_target_feat[task_name].reshape(nbr_samples, -1), axis=1)

            max_nbr_bins = 30
            bin_edges[task_name] = np.unique(np.sort(target_magnitudes)[np.linspace(0, len(target_magnitudes)-1, max_nbr_bins+1, dtype=int)])
            # bin_edges[task_name] = np.histogram_bin_edges(target_magnitudes, bins=30)
            bin_indices = np.digitize(target_magnitudes, bin_edges[task_name]) - 1

            nbr_bins = len(bin_edges[task_name]) - 1
            for bin_idx in range(nbr_bins):
                mask = bin_indices == bin_idx
                binned_signals[task_name][bin_idx] = interp_feat_error[task_name][mask]

        return bin_edges, binned_signals

    def plot_feature_error_against_target_magnitude(self, interp_target_feat, interp_feat_error, tag, mode, schemeset, step_index):
        bin_edges, binned_signals = self._group_feature_error_data_into_bins_wrt_target_magnitude(interp_target_feat, interp_feat_error)
        fig, axes_array = plt.subplots(
            nrows=len(binned_signals),
            ncols=2,
            figsize=[10, 3*len(self._configs.tasks)],
            squeeze=False,
            dpi=PYPLOT_DPI,
            tight_layout=True,
        )
        for j, task_name in enumerate(sorted(binned_signals.keys())):
            # MEAN
            axes_array[j,0].bar(
                bin_edges[task_name][:-1],
                np.array([feat_errors_in_bin.mean() for feat_errors_in_bin in binned_signals[task_name]]),
                np.diff(bin_edges[task_name]),
                align = 'edge',
            )
            axes_array[j,0].set_title(task_name)
            axes_array[j,0].set_xlabel('Target feature value')
            axes_array[j,0].set_ylabel('Feature error - mean')

            # STD
            axes_array[j,1].bar(
                bin_edges[task_name][:-1],
                np.array([feat_errors_in_bin.std() for feat_errors_in_bin in binned_signals[task_name]]),
                np.diff(bin_edges[task_name]),
                align = 'edge',
            )
            axes_array[j,1].set_title(task_name)
            axes_array[j,1].set_xlabel('Target feature value')
            axes_array[j,1].set_ylabel('Feature error - std')
        self._writer.add_figure('{}_{}/{}'.format(mode, tag, schemeset), fig, step_index)

    def plot_feature_histograms(self, signals, tag, mode, schemeset, step_index):
        fig, axes_array = plt.subplots(
            nrows=len(self._configs.tasks),
            ncols=len(signals),
            figsize=[np.clip(4*len(signals), 6, 12), 2*len(self._configs.tasks)],
            squeeze=False,
            dpi=PYPLOT_DPI,
            tight_layout=True,
        )

        for k, signal_name in enumerate(sorted(signals.keys())):
            for j, task_name in enumerate(sorted(self._configs.tasks.keys())):
                axes_array[j,k].hist(
                    signals[signal_name][task_name],
                    bins = 30,
                )
                axes_array[j,k].set_title('/\n'.join([signal_name, task_name]))

        self._writer.add_figure('{}_{}/{}'.format(mode, tag, schemeset), fig, step_index)

    def calc_and_plot_signal_stats(self, signals, mode, schemeset, step_index, target_prior_samples=None, filtered_flag=False):
        suffix = '' if not filtered_flag else '_filtered'
        self.plot_feature_error_against_target_magnitude(signals['interp_target_feat'+suffix], signals['interp_pred_feat'+suffix], 'pred_feature_against_target_magnitude'+suffix, mode, schemeset, step_index)
        self.plot_feature_error_against_target_magnitude(signals['interp_target_feat'+suffix], signals['relative_feat_abserror'+suffix], 'relative_feature_abserror_against_target_magnitude'+suffix, mode, schemeset, step_index)
        self.plot_feature_error_against_target_magnitude(signals['interp_target_feat'+suffix], signals['interp_feat_abserror'+suffix], 'feature_abserror_against_target_magnitude'+suffix, mode, schemeset, step_index)
        self.plot_feature_error_against_target_magnitude(signals['interp_target_feat'+suffix], signals['interp_feat_error'+suffix], 'feature_error_against_target_magnitude'+suffix, mode, schemeset, step_index)

        histogram_signals = [
            'interp_target_feat'+suffix,
            'interp_pred_feat'+suffix,
            'pred_feat_raw'+suffix,
        ]
        signal_dict = {signal_name: signal for signal_name, signal in signals.items() if signal_name in histogram_signals}
        if target_prior_samples is not None:
            signal_dict['target_prior_samples'] = target_prior_samples
        self.plot_feature_histograms(signal_dict, 'feature_histograms'+suffix, mode, schemeset, step_index)

    def _retrieve_input_img(self, image_tensor):
        img = normalize(image_tensor, mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        img = torch.clamp(img / 255., 0.0, 1.0)
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

        unit = self._configs.targets[self._configs.tasks[task_name]['target']]['unit']
        if unit == 'px':
            format_spec = '{:.2f}'
            unit_suffix = ' px'
        elif unit == 'angle':
            format_spec = '{:.1f}'
            unit_suffix = ' deg'
        elif unit == 'cosdist':
            format_spec = '{:.1f}'
            unit_suffix = ' deg'
        elif unit == 'log_factor':
            format_spec = '{:.2f}'
            unit_suffix = ' factor'
        else:
            return '{}'.format(feat)

        tmp = ', '.join([format_spec.format(self._human_interp_maps[task_name](val)) for val in feat])
        if len(feat) > 1:
            tmp = '(' + tmp + ')'
        return tmp + unit_suffix

    def save_images(self, img1_batch, img2_batch, ref_img_path_batch, pred_features, target_features, loss_notapplied, mode, schemeset, step_index, sample=-1):
        img_shape = img1_batch.shape[-2:]
        img1 = self._retrieve_input_img(img1_batch[sample])
        img2 = self._retrieve_input_img(img2_batch[sample])
        ref_img_path = ref_img_path_batch[sample]

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
        ref_img_title = 'Ref. image'
        if ref_img_path is not None:
            ref_img_title += '\n' + ref_img_path
        self._plot_img(axes_array[0,0], img1, ref_img_title)
        self._plot_img(axes_array[0,1], img2, 'Query image')

        def task_lines(task_name):
            lines = [
                task_name,
                '{:<8s}{:<8s} {:s}'.format('-', 'pred:', self._pretty_print_feature_value(task_name, pred_features[task_name][sample].detach().cpu().numpy())),
                '{:<8s}{:<8s} {:s}'.format('-', 'target:', self._pretty_print_feature_value(task_name, target_features[task_name][sample].detach().cpu().numpy())),
                '{:<8s}{:<8s} {:.1f}'.format('-', 'applied:', torch.mean((~loss_notapplied[task_name][sample]).float()).detach().cpu().numpy()),
            ]
            return lines
        lines = [line for task_name in sorted(self._configs.tasks.keys()) for line in task_lines(task_name)]
        text = '\n'.join(lines)

        gs = axes_array[1,0].get_gridspec() # Might as well have taken gridspec from any axes..?
        axes_array[1,0].remove()
        axes_array[1,1].remove()
        row1_ax = fig.add_subplot(gs[1, :])
        self._plot_text(row1_ax, text)

        self._writer.add_figure('{}/{}'.format(mode, schemeset), fig, step_index)

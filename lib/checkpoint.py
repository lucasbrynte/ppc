"""Checkpoint handler."""
import os
import logging
import torch

from lib.utils import get_device

class CheckpointHandler:
    """Save and load PyTorch checkpoint."""
    def __init__(self, configs):
        self._configs = configs
        self._logger = logging.getLogger(self.__class__.__name__)
        self._best_score = -float('Inf')
        self._checkpoint_dir = os.path.join(configs.experiment_path, 'checkpoints')
        if self._configs.train_or_eval in ['eval', 'eval_poseopt']:
            self._old_checkpoint_dir = os.path.join(configs.old_experiment_path, 'checkpoints')
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def init(self, model, force_load=False):
        """Create or load model."""
        model = model.to(get_device())
        if self._configs.train_or_eval in ['eval', 'eval_poseopt']:
            load_path = os.path.join(self._old_checkpoint_dir, self._configs.checkpoint_load_fname)
            self._logger.info('Loading checkpoint from: %s', load_path)
            checkpoint = torch.load(load_path, map_location=get_device())
            model.load_state_dict(checkpoint)
        return model

    def save(self, model, epoch, score):
        state_dict = model.state_dict()
        if score > self._best_score:
            self._best_score = score
            file_name = 'best_model.pth.tar'
            torch.save(state_dict, os.path.join(self._checkpoint_dir, file_name))
        if self._configs.training.backup_epochs:
            file_name = 'epoch{0:03d}.pth.tar'.format(epoch)
            torch.save(state_dict, os.path.join(self._checkpoint_dir, file_name))
        # Always save latest
        torch.save(state_dict, os.path.join(self._checkpoint_dir, 'latest_model.pth.tar'))

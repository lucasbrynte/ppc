"""Reading input data in the SIXD common format."""
from collections import namedtuple
import os
import shutil

import numpy as np
# from PIL import Image
import torch
from torch import Tensor
# from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset

from lib.constants import TRAIN, VAL
from lib.loader import Sample

def get_dataset(configs, mode):
    return DummyDataset(configs, mode)


Annotation = namedtuple('Annotation', ['delta_rot', 'delta_loc'])


class DummyDataset(Dataset):
    def __init__(self, configs, mode):
        self._configs = configs
        self._mode = mode
        self._aug_transform = None
        # if self._mode == TRAIN:
        #     self._aug_transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03)
        #     # self._aug_transform = ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5), saturation=(0.7, 1.5), hue=(-0.03, 0.03))
        # else:
        #     self._aug_transform = None

        self._pids_path = '/tmp/sixd_kp_pids/' + self._mode
        if os.path.exists(self._pids_path):
            shutil.rmtree(self._pids_path)
            print("Removing " + self._pids_path)
        print("Creating " + self._pids_path)
        os.makedirs(self._pids_path)

    def __len__(self):
        return 512

    def _init_worker_seed(self):
        pid = os.getpid()
        pid_path = os.path.join(self._pids_path, str(pid))
        if os.path.exists(pid_path):
            return
        np.random.seed(pid)
        open(pid_path, 'w').close()

    def __getitem__(self, index):
        self._init_worker_seed() # Cannot be called in constructor, since it is only executed by main process. Workaround: call at every sampling.
        data, annotations = self._generate_sample()
        return Sample(annotations, data)

    def _generate_sample(self):
        # TODO:
        # Sample R1, R2, t1, t2
        # Compute delta_rot & delta_loc
        # Render images, and stack into data pytorch tensor

        # Dummy data:
        img1 = torch.zeros([3] + list(self._configs.data.img_dims))
        img2 = torch.ones([3] + list(self._configs.data.img_dims))
        data = img1, img2

        delta_rot = torch.eye(3)
        delta_loc = torch.tensor([0.0, 1.0, 0.0])

        annotation = Annotation(
            delta_rot = delta_rot,
            delta_loc = delta_loc,
        )

        return data, annotation

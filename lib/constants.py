"""Constants for 3DOD."""
import os
import numpy as np

# Execution modes
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# Paths
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
CONFIG_ROOT = os.path.join(PROJECT_PATH, 'conf')

# TorchVision
TV_MEAN = np.array((0.485, 0.456, 0.406))
TV_STD = np.array((0.229, 0.224, 0.225))

# Matplotlib
PYPLOT_DPI = 100

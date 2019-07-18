"""Parsing input arguments."""
import argparse
import logging
import os
import yaml
import shutil
import torch

from lib.constants import PROJECT_PATH, TRAIN, VAL, TEST, CONFIG_ROOT
from lib.utils import show_gpu_info


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='3D object detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-name',
                        help='name of the config dir that is going to be used')
    parser.add_argument('--checkpoint-load-path', default='',
                        help='path of the model weights to load')
    parser.add_argument('--experiment-root-path', default=get_default_root(),
                        help='the root directory to hold experiments')
    parser.add_argument('--overwrite-experiment', action='store_true', default=False,
                        help='causes experiment to be overwritten, if it already exists')
    parser.add_argument('--experiment-name', default='3dod_demo',
                        help='name of the execution, will be '
                             'the name of the experiment\'s directory')
    # parser.add_argument('--eval-mode', action='append', default=[], type=str,
    #                     help='For eval.py only. Example: "--eval-mode val --eval-mode train" performs evaluation on train & val sets, val set first.')
    # parser.add_argument('--train-seqs', default=None, type=str)
    # parser.add_argument('--group-labels', default=None, type=str)


    args = parser.parse_args()

    experiment_path = os.path.join(args.experiment_root_path, args.experiment_name)
    args.experiment_path = experiment_path
    if args.overwrite_experiment and os.path.exists(args.experiment_path):
        shutil.rmtree(args.experiment_path)
    args.checkpoint_root_dir = os.path.join(experiment_path, 'checkpoints')
    os.makedirs(args.checkpoint_root_dir, exist_ok=True)

    return args


def get_default_root():
    """Get default root."""
    project_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(project_root_path, 'experiments')


def setup_logging(experiment_path, mode):
    """Setup logging."""
    logs_path = os.path.join(experiment_path, 'logs')
    log_file_name = '{}.log'.format(mode)
    os.makedirs(logs_path, exist_ok=True)
    log_path = os.path.join(logs_path, log_file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(fmt='%(levelname)-5s %(name)-10s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.info('Log file is %s', log_path)


def prepare_environment():
    """Prepare environment."""
    os.environ['TORCH_HOME'] = os.path.join(PROJECT_PATH, '.torch') # Where to save torch files, e.g. pretrained models from the web.
    cuda_is_available = torch.cuda.is_available()
    logging.info('Use cuda: %s', cuda_is_available)
    if cuda_is_available:
        show_gpu_info()
        torch.backends.cudnn.benchmark = True # Boost if input size constant. https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2


def save_settings(args):
    """Save user settings to experiment's setting directory."""
    experiment_settings_path = os.path.join(args.experiment_path, 'settings')
    logging.info('Save settings to %s', experiment_settings_path)

    shutil.rmtree(experiment_settings_path, ignore_errors=True)
    shutil.copytree(os.path.join(CONFIG_ROOT, args.config_name), experiment_settings_path)
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'default_config.yml'),
                    os.path.join(experiment_settings_path, 'default_config.yml'))

    with open(os.path.join(experiment_settings_path, 'args.yml'), 'w') as file:
        yaml.dump(vars(args), file, Dumper=yaml.CDumper)

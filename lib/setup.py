"""Parsing input arguments."""
import argparse
import logging
import os
import yaml
import shutil
import torch

from lib.constants import PROJECT_PATH, TRAIN, VAL, TEST, CONFIG_ROOT
from lib.utils import show_gpu_info
from lib.utils import read_attrdict_from_default_and_specific_yaml


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Pose Proposal Critic',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_or_eval', choices=['train', 'eval'],
                        help='select train / eval mode')
    parser.add_argument('--config-name',
                        help='name of the config dir that is going to be used')
    parser.add_argument('--experiment-root-path', default=get_default_root(),
                        help='the root directory to hold experiments')
    parser.add_argument('--overwrite-experiment', action='store_true', default=False,
                        help='causes experiment to be overwritten, if it already exists')
    parser.add_argument('--experiment-name', required=True, type=str,
                        help='name of the execution, will be '
                             'the name of the experiment\'s directory')
    parser.add_argument('--old-experiment-name', default=None, type=str,
                        help='name of experiment to evaluate')
    parser.add_argument('--checkpoint-load-fname', default='best_model.pth.tar',
                        help='file name of the model weights to load before evaluation')
    parser.add_argument('--eval-mode', action='append', default=[], type=str,
                        help='For eval only. Example: "--eval-mode val --eval-mode train" performs evaluation on train & val sets, val set first.')
    # parser.add_argument('--train-seqs', default=None, type=str)
    parser.add_argument('--obj-label', required=True, type=str)


    args = parser.parse_args()

    if args.train_or_eval == 'eval':
        assert args.old_experiment_name is not None
        assert len(args.eval_mode) > 0
    else:
        assert args.old_experiment_name is None
        assert args.eval_mode == []

    args.experiment_path = os.path.join(args.experiment_root_path, args.experiment_name)
    if args.train_or_eval == 'eval':
        args.old_experiment_path = os.path.join(args.old_experiment_root_path, args.old_experiment_name)

    if args.overwrite_experiment and os.path.exists(args.experiment_path):
        shutil.rmtree(args.experiment_path)

    args.checkpoint_root_dir = os.path.join(args.experiment_path, 'checkpoints')
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
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'default_setup.yml'),
                    os.path.join(experiment_settings_path, 'default_setup.yml'))
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'default_runtime.yml'),
                    os.path.join(experiment_settings_path, 'default_runtime.yml'))

    with open(os.path.join(experiment_settings_path, 'args.yml'), 'w') as file:
        yaml.dump(vars(args), file, Dumper=yaml.CDumper)

def get_configs(args):
    if args.train_or_eval == 'train':
        # Read from configuration
        configs = read_attrdict_from_default_and_specific_yaml(
            os.path.join(CONFIG_ROOT, 'default_setup.yml'),
            os.path.join(CONFIG_ROOT, args.config_name, 'setup.yml'),
        )
    else:
        # Read from old experiment
        old_experiment_settings_path = os.path.join(args.old_experiment_path, 'settings')
        configs = read_attrdict_from_default_and_specific_yaml(
            os.path.join(old_experiment_settings_path, 'default_setup.yml'),
            os.path.join(old_experiment_settings_path, args.config_name, 'setup.yml'),
        )

    configs.runtime = read_attrdict_from_default_and_specific_yaml(
        os.path.join(CONFIG_ROOT, 'default_runtime.yml'),
        os.path.join(CONFIG_ROOT, args.config_name, 'runtime.yml'),
    )

    configs += vars(args)

    if args.train_or_eval == 'eval':
        # NOTE: The loading options for TEST is used also for TRAIN & VAL during evaluation.
        configs['loading'][TRAIN]['shuffle'] = configs['loading'][TEST]['shuffle']
        configs['loading'][VAL]['shuffle'] = configs['loading'][TEST]['shuffle']
        # configs['loading'][TRAIN]['max_nbr_batches'] = configs['loading'][TEST]['max_nbr_batches']
        # configs['loading'][VAL]['max_nbr_batches'] = configs['loading'][TEST]['max_nbr_batches']

    return configs

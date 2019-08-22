"""Parsing input arguments."""
import argparse
import logging
import os
import ruamel.yaml as yaml
import shutil
import torch
from attrdict import AttrDict
import numpy as np

from lib.constants import PROJECT_PATH, TRAIN, VAL, TEST, CONFIG_ROOT
from lib.utils import show_gpu_info, closeto_within
from lib.utils import read_yaml_as_attrdict, read_attrdict_from_default_and_specific_yaml


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
    # parser.add_argument('--eval-mode', action='append', default=[], type=str,
    #                     help='For eval only. Example: "--eval-mode val --eval-mode train" performs evaluation on train & val sets, val set first.')
    # parser.add_argument('--train-seqs', default=None, type=str)
    parser.add_argument('--obj-label', required=True, type=str)


    args = parser.parse_args()

    if args.train_or_eval == 'eval':
        assert args.old_experiment_name is not None
        # assert len(args.eval_mode) > 0
    else:
        assert args.old_experiment_name is None
        # assert args.eval_mode == []

    args.experiment_path = os.path.join(args.experiment_root_path, args.experiment_name)
    if args.train_or_eval == 'eval':
        args.old_experiment_path = os.path.join(args.experiment_root_path, args.old_experiment_name)

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
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'default_runtime_train.yml'),
                    os.path.join(experiment_settings_path, 'default_runtime_train.yml'))
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'default_runtime_eval.yml'),
                    os.path.join(experiment_settings_path, 'default_runtime_eval.yml'))
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'ref_sampling_schemes.yml'),
                    os.path.join(experiment_settings_path, 'ref_sampling_schemes.yml'))
    shutil.copyfile(os.path.join(CONFIG_ROOT, 'query_sampling_schemes.yml'),
                    os.path.join(experiment_settings_path, 'query_sampling_schemes.yml'))

    with open(os.path.join(experiment_settings_path, 'args.yml'), 'w') as file:
        yaml.dump(vars(args), file, Dumper=yaml.CDumper)

def infer_sampling_probs(sampling_scheme_list):
    nbr_schemes = len(sampling_scheme_list)
    nbr_schemes_with_unspecified_prob = sum(['sampling_prob' not in sampling_scheme_def for sampling_scheme_def in sampling_scheme_list])
    total_prob_specified = sum([sampling_scheme_def['sampling_prob'] for sampling_scheme_def in sampling_scheme_list if 'sampling_prob' in sampling_scheme_def])
    assert closeto_within(total_prob_specified, low=0.0, high=1.0)
    remaining_prob = 1.0 - total_prob_specified
    for sampling_scheme_def in sampling_scheme_list:
        if 'sampling_prob' not in sampling_scheme_def:
            sampling_scheme_def['sampling_prob'] = remaining_prob / nbr_schemes_with_unspecified_prob
    total_prob = sum([sampling_scheme_def['sampling_prob'] for sampling_scheme_def in sampling_scheme_list])
    # Should now sum up to 1.0:
    assert np.isclose(total_prob, 1.0)
    # NOTE: Not returning sampling_scheme_list, in order to emphasize in-place behavior

def get_configs(args):
    if args.train_or_eval == 'train':
        # Read from configuration
        configs = read_attrdict_from_default_and_specific_yaml(
            os.path.join(CONFIG_ROOT, 'default_setup.yml'),
            os.path.join(CONFIG_ROOT, args.config_name, 'setup.yml'),
        )
        configs.runtime = read_attrdict_from_default_and_specific_yaml(
            os.path.join(CONFIG_ROOT, 'default_runtime_train.yml'),
            os.path.join(CONFIG_ROOT, args.config_name, 'runtime_train.yml'),
        )
    else:
        # Read from old experiment
        old_experiment_settings_path = os.path.join(args.old_experiment_path, 'settings')
        configs = read_attrdict_from_default_and_specific_yaml(
            os.path.join(old_experiment_settings_path, 'default_setup.yml'),
            os.path.join(old_experiment_settings_path, args.config_name, 'setup.yml'),
        )
        configs.runtime = read_attrdict_from_default_and_specific_yaml(
            os.path.join(CONFIG_ROOT, 'default_runtime_eval.yml'),
            os.path.join(CONFIG_ROOT, args.config_name, 'runtime_eval.yml'),
        )

    if args.train_or_eval == 'train':
        assert tuple(configs['runtime']['data_sampling_scheme_defs'].keys()) == (TRAIN, VAL)
        # Validate config: Only VAL / TEST support multiple runs with different sets of sampling schemes
        assert tuple(configs['runtime']['data_sampling_scheme_defs'][TRAIN].keys()) == (TRAIN,), 'For training, there may only be a single set of sampling schemes to sample from.'
    else:
        assert tuple(configs['runtime']['data_sampling_scheme_defs'].keys()) == (TEST,)


    modes = (TRAIN, VAL) if args.train_or_eval == 'train' else (TEST,)

    # ==========================================================================
    # Apply default options to all schemeset defs
    # ==========================================================================
    for mode in modes:
        default_schemeset_opts = getattr(configs.runtime.default_schemeset_opts, mode)
        for schemeset, schemeset_def in configs['runtime']['data_sampling_scheme_defs'][mode].items():
            if 'opts' in schemeset_def:
                override_opts = AttrDict(schemeset_def['opts'])
                schemeset_def['opts'] = default_schemeset_opts + override_opts
            else:
                schemeset_def['opts'] = default_schemeset_opts

    # ==========================================================================
    # Determine choice of data sampling specs for each mode, and store them in config
    # ==========================================================================
    all_ref_sampling_schemes = read_yaml_as_attrdict(os.path.join(CONFIG_ROOT, 'ref_sampling_schemes.yml'))
    all_query_sampling_schemes = read_yaml_as_attrdict(os.path.join(CONFIG_ROOT, 'query_sampling_schemes.yml'))
    ref_sampling_schemes = {}
    query_sampling_schemes = {}
    for mode in modes:
        ref_sampling_schemes[mode] = {}
        query_sampling_schemes[mode] = {}
        for scheme_set_name in configs['runtime']['data_sampling_scheme_defs'][mode].keys():
            ref_sampling_scheme_list = configs['runtime']['data_sampling_scheme_defs'][mode][scheme_set_name]['ref_schemeset'] # List of elements such as {ref_scheme: real_unoccl_train}
            query_sampling_scheme_list = configs['runtime']['data_sampling_scheme_defs'][mode][scheme_set_name]['query_schemeset'] # List of elements such as {query_scheme: rot_only_20deg_std}
            infer_sampling_probs(ref_sampling_scheme_list) # Modified in-place
            if configs.runtime.data_sampling_scheme_defs[mode][scheme_set_name]['opts']['loading']['coupled_ref_and_query_scheme_sampling']:
                # Ref & query schemes are sampled jointly. Lists need to be of same length to be able to map elements.
                assert len(query_sampling_scheme_list) == len(ref_sampling_scheme_list)
            else:
                infer_sampling_probs(query_sampling_scheme_list) # Modified in-place
            ref_sampling_schemes[mode][scheme_set_name] = [all_ref_sampling_schemes[ref_sampling_scheme_def['ref_scheme']] for ref_sampling_scheme_def in ref_sampling_scheme_list] # Map all such elements to the corresponding data sampling specs
            query_sampling_schemes[mode][scheme_set_name] = [all_query_sampling_schemes[query_sampling_scheme_def['query_scheme']] for query_sampling_scheme_def in query_sampling_scheme_list] # Map all such elements to the corresponding data sampling specs
    configs['runtime']['ref_sampling_schemes'] = AttrDict(ref_sampling_schemes)
    configs['runtime']['query_sampling_schemes'] = AttrDict(query_sampling_schemes)


    configs += vars(args)

    return configs

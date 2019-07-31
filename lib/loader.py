"""Load batches for training."""
from collections import namedtuple
from importlib import import_module
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

Sample = namedtuple('Sample', ['targets', 'input', 'extra_input'])
Batch = namedtuple('Batch', ['targets', 'input', 'extra_input'])


class FixedSeededRandomSampler(RandomSampler):
    """
    Tweak RandomSampler to:
        Sample an epoch once, and iterate in this order always.
        Use a random seed for sampling.
    """
    def __init__(self, *args, seed='314159', **kwargs):
        super().__init__(*args, **kwargs)

        # Set random seed
        if seed is not None:
            self._set_seed(seed)

        # Sample an epoch as usual with RandomSampler, but store for re-use
        self._fixed_idx_list = list(super().__iter__())

        # Reset RNG state
        if seed is not None:
            self._reset_rng_state()

    def __iter__(self):
        for idx in self._fixed_idx_list:
            yield idx

    def _set_seed(self, seed):
        self._rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    def _reset_rng_state(self):
        torch.set_rng_state(self._rng_state)

# class Loader:
#     """docstring for Loader."""
#     def __init__(self, modes, configs):
#         self._configs = configs
#         self._dataset_module = import_module('lib.datasets.%s' % configs.data.dataformat)
#         for mode in modes:
#             dataset = self._dataset_module.get_dataset(self._configs, mode)
#             setattr(self, mode, dataset)
# 
#     def gen_batches(self, mode):
#         """Return an iterator over batches."""
#         # getattr(self, mode).__getitem__(0)
#         # assert False
# 
#         assert getattr(self._configs.loading, mode).batch_size == 1
#         for sample in getattr(self, mode):
#             batch = collate_batch([sample])
#             batch = Batch(*(val if fieldname != 'targets' else getattr(self, mode).dataset.Targets(*val) for fieldname, val in zip(Batch._fields, batch)))
#             yield batch


class Loader:
    """docstring for Loader."""
    def __init__(self, modes, configs):
        self._configs = configs
        self._dataset_module = import_module('lib.datasets.%s' % configs.data.dataformat)
        self._datasets = {mode: self._dataset_module.get_dataset(self._configs, mode) for mode in modes}

    def _init_loader(self, mode, nbr_samples):
        loader_configs = self._get_loader_config(mode, nbr_samples)
        loader = DataLoader(**loader_configs)
        return loader

    def _get_loader_config(self, mode, nbr_samples):
        self._datasets[mode].set_len(nbr_samples)
        data_configs = getattr(self._configs.loading, mode)
        loader_config = {}
        loader_config['dataset'] = self._datasets[mode]
        loader_config['collate_fn'] = collate_batch
        loader_config['pin_memory'] = True
        loader_config['batch_size'] = data_configs.batch_size
        # loader_config['num_workers'] = data_configs.num_workers
        loader_config['drop_last'] = True
        if data_configs.shuffle == True:
            loader_config['sampler'] = RandomSampler(self._datasets[mode])
        elif data_configs.shuffle == False:
            loader_config['sampler'] = SequentialSampler(self._datasets[mode])
        elif data_configs.shuffle == 'fixed':
            loader_config['sampler'] = FixedSeededRandomSampler(self._datasets[mode], seed='314159')
        else:
            # Should not happen
            assert False
        return loader_config

    def gen_batches(self, mode, nbr_samples):
        """Return an iterator over batches."""
        loader = self._init_loader(mode, nbr_samples)
        # NOTE: Seems to be fixed in pytorch 1.1.0? What version used in container?
        # TODO: This is needed until pytorch pin_memory is fixed. Currently casts namedtuple to list
        # https://github.com/pytorch/pytorch/pull/16440
        for batch in loader:
            batch = Batch(
                targets = loader.dataset.Targets(*batch[0]),
                input = batch[1],
                extra_input = self._dataset_module.ExtraInput(*batch[2]),
            )
            yield batch


def collate_batch(batch_list):
    """Collates for PT data loader."""
    targets, in_data, extra_input = zip(*batch_list)

    # Map list hierarchy from sample/property to property/sample
    targets = tuple(map(torch.stack, zip(*targets)))
    extra_input = tuple(map(torch.stack, zip(*extra_input)))

    img1_batch, img2_batch = zip(*in_data)
    in_data = torch.stack(img1_batch), torch.stack(img2_batch)
    return (targets, in_data, extra_input)

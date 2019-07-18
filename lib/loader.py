# RENDER IMGS FROM SLIGHTLY DIFFERENT VIEWS WITH GLUMPY
# Transl error only initially
# Use dict instead of attrdict for annotations. No implicit data type conversions - feels more controlled
# Render / extract 128x128 patches

"""Load batches for training."""
from collections import namedtuple
from importlib import import_module
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

Sample = namedtuple('Sample', ['annotation', 'input'])
Batch = namedtuple('Batch', ['annotation', 'input'])


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

class Loader:
    """docstring for Loader."""
    def __init__(self, modes, configs):
        self._configs = configs
        self._dataset_module = import_module('lib.datasets.%s' % configs.data.dataformat)
        for mode in modes:
            loader_configs = self._get_loader_config(mode)
            loader = DataLoader(**loader_configs)
            setattr(self, mode, loader)

    def _get_loader_config(self, mode):
        dataset = self._dataset_module.get_dataset(self._configs, mode)
        data_configs = getattr(self._configs.loading, mode)
        loader_config = {}
        loader_config['dataset'] = dataset
        loader_config['collate_fn'] = collate_batch
        loader_config['pin_memory'] = True
        loader_config['batch_size'] = data_configs.batch_size
        # loader_config['num_workers'] = data_configs.num_workers
        loader_config['drop_last'] = True
        if data_configs.shuffle == True:
            loader_config['sampler'] = RandomSampler(dataset)
        elif data_configs.shuffle == False:
            loader_config['sampler'] = SequentialSampler(dataset)
        elif data_configs.shuffle == 'fixed':
            loader_config['sampler'] = FixedSeededRandomSampler(dataset, seed='314159')
        else:
            # Should not happen
            assert False
        return loader_config

    def gen_batches(self, mode):
        """Return an iterator over batches."""
        # NOTE: Seems to be fixed in pytorch 1.1.0? What version used in container?
        # TODO: This is needed until pytorch pin_memory is fixed. Currently casts namedtuple to list
        # https://github.com/pytorch/pytorch/pull/16440
        for batch in getattr(self, mode):
            batch = Batch(*(val if fieldname != 'annotation' else self._dataset_module.Annotation(*val) for fieldname, val in zip(Batch._fields, batch)))
            yield batch


def collate_batch(batch_list):
    """Collates for PT data loader."""
    annotations, in_data = zip(*batch_list)

    # Map list hierarchy from sample/property to property/sample
    annotations = tuple(map(torch.stack, zip(*annotations)))

    img1_batch, img2_batch = zip(*in_data)
    in_data = torch.stack(img1_batch), torch.stack(img2_batch)
    return (annotations, in_data)

#     def gen_batches(self, mode):
#         """Return an iterator over batches."""
#         # NOTE: Seems to be fixed in pytorch 1.1.0? What version used in container?
#         # TODO: This is needed until pytorch pin_memory is fixed. Currently casts namedtuple to list
#         # https://github.com/pytorch/pytorch/pull/16440
#         for batch in getattr(self, mode):
#             batch = Batch(*batch)
#             for annotation in batch.annotation:
#                 annotation = self._dataset_module.Annotation(*annotation)
#             # for annotations in batch.annotation:
#             #     for index, annotation in enumerate(annotations):
#             #         annotations[index] = self._dataset_module.Annotation(*annotation)
#             yield batch
# 
# 
# def collate_batch(batch_list):
#     """Collates for PT data loader."""
#     annotations, in_data = zip(*batch_list)
#     # in_data = torch.stack(in_data)
#     img1_batch, img2_batch = zip(*in_data)
#     in_data = torch.stack(img1_batch), torch.stack(img2_batch)
#     return (annotations, in_data)

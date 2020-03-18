from attrdict import AttrDict
import numpy as np
import torch
from torch import nn
# from torchvision.models import resnet18 as r18
from lib.models.resnet_tv050 import resnet18 as r18
from lib.models.resnet_tv050 import resnet34 as r34
from lib.models.common import Head
from lib.utils import get_module_parameters, get_device


class Resnet18Wrapper(nn.Module):
    def __init__(self, configs, pretrained=False, include_lowlevel=True, include_midlevel=True, include_highlevel=True, depth=18, init_ds=True, max_pool=True, final_ds=None, avg_pool=False):
        super().__init__()
        self._configs = configs
        self._include_lowlevel = include_lowlevel
        self._include_midlevel = include_midlevel
        self._include_highlevel = include_highlevel
        self._init_ds = init_ds
        self._max_pool = max_pool
        self._final_ds = final_ds
        self._avg_pool = avg_pool
        assert self._include_lowlevel or self._include_midlevel or self._include_highlevel
        if self._include_lowlevel and self._include_highlevel:
            assert self._include_midlevel

        if depth == 18:
            resnet18 = r18(
                pretrained = pretrained,
                init_ds = self._init_ds,
                max_pool = self._max_pool,
            )
        else:
            assert depth == 34
            resnet18 = r34(
                pretrained = pretrained,
                init_ds = self._init_ds,
                max_pool = self._max_pool,
            )

        if self._include_lowlevel:
            self.conv1 = resnet18.conv1
            self.bn1 = resnet18.bn1
            self.relu = resnet18.relu
            self.max_pool = resnet18.max_pool
            self.layer1 = resnet18.layer1

        if self._include_midlevel:
            self.layer2 = resnet18.layer2
            self.layer3 = resnet18.layer3

        if self._include_highlevel:
            self.layer4 = resnet18.layer4

        if self._avg_pool:
            self.avgpool = resnet18.avgpool
        # self.fc = resnet18.fc

    def forward(self, x):
        if self._include_lowlevel:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.layer1(x)

        if self._include_midlevel:
            x = self.layer2(x)
            x = self.layer3(x)

        if self._include_highlevel:
            x = self.layer4(x)

        if self._final_ds is not None:
            x = nn.functional.avg_pool2d(x, self._final_ds, self._final_ds)

        if self._avg_pool:
            x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self.verify_head_layer_specs(self._configs['model']['head_layer_specs']) # Access by key important - needs to be mutable

        # Embedding modules as referred to in DPOD paper:
        self.E11 = Resnet18Wrapper(
            self._configs,
            pretrained = self._configs.model.resnet18_opts.pretrained,
            include_lowlevel = True,
            include_midlevel = False,
            include_highlevel = False,
            depth = self._configs.model.resnet18_opts.depth,
            init_ds = self._configs.model.resnet18_opts.init_ds,
            max_pool = self._configs.model.resnet18_opts.max_pool,
            final_ds = None,
            avg_pool = False,
        )
        for param in self.E11.parameters():
            param.requires_grad = False
        if self._configs.model.resnet18_opts.unfreeze_merge_layer:
            for param in self.E11.layer1[-1].conv2.parameters():
                param.requires_grad = True
        # # NOTE: try/catch for backwards compatibility
        # try:
        #     if self._configs.model.resnet18_opts.unfreeze_merge_layer:
        #         for param in self.E11.layer1[-1].conv2.parameters():
        #             param.requires_grad = True
        # except AttributeError:
        #     pass
        self.E12 = Resnet18Wrapper(
            self._configs,
            pretrained = self._configs.model.resnet18_opts.pretrained,
            include_lowlevel = True,
            include_midlevel = False,
            include_highlevel = False,
            depth = self._configs.model.resnet18_opts.depth,
            init_ds = self._configs.model.resnet18_opts.init_ds,
            max_pool = self._configs.model.resnet18_opts.max_pool,
            final_ds = None,
            avg_pool = False,
        )
        self.E2 = Resnet18Wrapper(
            self._configs,
            pretrained = self._configs.model.resnet18_opts.pretrained and self._configs.model.resnet18_opts.E2_pretrained,
            include_lowlevel = False,
            include_midlevel = True,
            include_highlevel = self._configs.model.resnet18_opts.E2_include_highlevel,
            depth = self._configs.model.resnet18_opts.depth,
            init_ds = self._configs.model.resnet18_opts.init_ds,
            max_pool = self._configs.model.resnet18_opts.max_pool,
            final_ds = self._configs.model.resnet18_opts.final_ds,
            avg_pool = self._configs.model.resnet18_opts.avg_pool,
        )

        resnet_output_dims = self._check_resnet_output_dims()

        self.heads = nn.ModuleDict({ head_name: Head(self._configs, resnet_output_dims, list(map(AttrDict, head_spec['layers']))) for head_name, head_spec in self._configs.model.head_layer_specs.items() })

    def freeze_resnet(self):
        for param in list(self.E12.parameters()) + list(self.E2.parameters()):
            param.requires_grad = False

    def unfreeze_resnet(self):
        for param in list(self.E12.parameters()) + list(self.E2.parameters()):
            param.requires_grad = True

    def _check_resnet_output_dims(self):
        out = self.E2(self.E11(torch.zeros((1, 3, self._configs.data.crop_dims[0], self._configs.data.crop_dims[1]))))
        return out.shape[1:]

    def get_optim_param_groups(self):
        weight_decay = self._configs.model.resnet18_opts.weight_decay if self._configs.model.resnet18_opts.weight_decay is not None else 0.0
        param_groups = []
        param_groups.append({
            'params': [param for name, param in self.named_parameters() if name.endswith('weight') and param.requires_grad],
            'weight_decay': self._configs.model.resnet18_opts.weight_decay,
        })
        param_groups.append({
            'params': [param for name, param in self.named_parameters() if name.endswith('bias') and param.requires_grad],
            'weight_decay': 0,
        })
        nbr_params = sum([len(param_group['params']) for param_group in param_groups])
        total_nbr_params = len([param for param in self.parameters() if param.requires_grad])
        assert nbr_params == total_nbr_params
        return param_groups

    def verify_head_layer_specs(self, all_head_specs):
        for head_name, head_spec in all_head_specs.items():
            if head_spec['tasks'] is None:
                head_spec['tasks'] = sorted(self._configs.tasks.keys())
            n_out = sum([self._configs.targets[self._configs.tasks[task_name]['target']]['n_out'] for task_name in head_spec['tasks']])
            if 'n_out' in head_spec['layers'][-1]:
                assert head_spec['layers'][-1]['n_out'] == n_out
            else:
                head_spec['layers'][-1]['n_out'] = n_out
            assert head_spec['layers'][-1]['relu_flag'] == False

    def forward(self, ref_img, query_img):
        x = self.E11(ref_img) - self.E12(query_img)
        x = self.E2(x)
        x = { head_name: self.heads[head_name](x) for head_name in self.heads }
        # x = self.head(x)
        return {
            'features': x,
        }

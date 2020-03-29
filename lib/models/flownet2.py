from attrdict import AttrDict
import numpy as np
import torch
from torch import nn
from lib.models.flownet2_resources.FlowNetS import FlowNetS
from lib.models.flownet2_resources.FlowNetSD import FlowNetSD
from lib.models.flownet2_resources.submodules import predict_mask
from lib.models.common import Head
from lib.utils import get_module_parameters, get_device

# from torchvision.transforms.functional import normalize
from lib.constants import TV_MEAN, TV_STD

def renormalize_img(img):
    """
    Map pixel intensities from normalized to [-0.5, 0.5] range.
    """
    img = img * TV_STD[None,:,None,None].to(img.device)
    img = img + TV_MEAN[None,:,None,None].to(img.device)
    img = img / 255.
    img = img - 0.5
    return img

class FlowNetS_wrapper(FlowNetS):
    def __init__(self, configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configs = configs

        self.predict_mask4 = predict_mask(770)
        self.upsampled_flow4_to_0 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsampled_mask4_to_0 = nn.Upsample(scale_factor=16, mode='nearest')
        self.predict_mask2 = predict_mask(194)
        self.upsampled_mask2_to_0 = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, ref_img, query_img):
        if self._configs.model.flownet2_opts.renormalize_imgs:
            ref_img = renormalize_img(ref_img)
            query_img = renormalize_img(query_img)
        ref_img = ref_img * self._configs.model.flownet2_opts.rescale_img_factor
        query_img = query_img * self._configs.model.flownet2_opts.rescale_img_factor

        x = torch.cat((ref_img, query_img), dim=1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        with torch.set_grad_enabled(self._configs.aux_tasks.fg_mask.enabled):
            flow6       = self.predict_flow6(out_conv6)
            flow6_up    = self.upsampled_flow6_to_5(flow6)
            out_deconv5 = self.deconv5(out_conv6)
            
            concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
            flow5       = self.predict_flow5(concat5)
            flow5_up    = self.upsampled_flow5_to_4(flow5)
            out_deconv4 = self.deconv4(concat5)
            
            concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
            flow4       = self.predict_flow4(concat4)

            x = out_conv6
            flow = self.upsampled_mask4_to_0(flow4)
            fg_mask4 = self.predict_mask4(concat4)
            fg_mask = self.upsampled_mask4_to_0(fg_mask4)
        return x, fg_mask, flow

        #     flow4_up    = self.upsampled_flow4_to_3(flow4)
        #     out_deconv3 = self.deconv3(concat4)
        # 
        #     concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        #     flow3       = self.predict_flow3(concat3)
        #     flow3_up    = self.upsampled_flow3_to_2(flow3)
        #     out_deconv2 = self.deconv2(concat3)
        # 
        #     concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        #     flow2 = self.predict_flow2(concat2)
        # 
        # # NOTE: Put aux loss on flow on all levels?
        # # if self.training:
        # #     return flow2,flow3,flow4,flow5,flow6
        # # else:
        # #     return flow2,
        # 
        # x = out_conv6
        # 
        # with torch.set_grad_enabled(self._configs.aux_tasks.fg_mask):
        #     flow = self.upsample1(flow2)
        #     fg_mask2 = self.predict_mask2(concat2)
        #     fg_mask = self.upsampled_mask2_to_0(fg_mask2)
        # 
        # return x, fg_mask, flow

class FlowNetSD_wrapper(FlowNetSD):
    def __init__(self, configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configs = configs

        self.predict_mask4 = predict_mask(256)
        self.upsampled_flow4_to_0 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsampled_mask4_to_0 = nn.Upsample(scale_factor=16, mode='nearest')
        self.predict_mask2 = predict_mask(64)
        self.upsampled_mask2_to_0 = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, ref_img, query_img):
        if self._configs.model.flownet2_opts.renormalize_imgs:
            ref_img = renormalize_img(ref_img)
            query_img = renormalize_img(query_img)
        ref_img = ref_img * self._configs.model.flownet2_opts.rescale_img_factor
        query_img = query_img * self._configs.model.flownet2_opts.rescale_img_factor

        x = torch.cat((ref_img, query_img), dim=1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        with torch.set_grad_enabled(self._configs.aux_tasks.fg_mask):
            flow6       = self.predict_flow6(out_conv6)
            flow6_up    = self.upsampled_flow6_to_5(flow6)
            out_deconv5 = self.deconv5(out_conv6)
            
            concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
            out_interconv5 = self.inter_conv5(concat5)
            flow5       = self.predict_flow5(out_interconv5)

            flow5_up    = self.upsampled_flow5_to_4(flow5)
            out_deconv4 = self.deconv4(concat5)
            
            concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
            out_interconv4 = self.inter_conv4(concat4)
            flow4       = self.predict_flow4(out_interconv4)

            x = out_conv6
            flow = self.upsampled_mask4_to_0(flow4)
            fg_mask4 = self.predict_mask4(out_interconv4)
            fg_mask = self.upsampled_mask4_to_0(fg_mask4)
        return x, fg_mask, flow

        #     flow4_up    = self.upsampled_flow4_to_3(flow4)
        #     out_deconv3 = self.deconv3(concat4)
        # 
        #     concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        #     out_interconv3 = self.inter_conv3(concat3)
        #     flow3       = self.predict_flow3(out_interconv3)
        #     flow3_up    = self.upsampled_flow3_to_2(flow3)
        #     out_deconv2 = self.deconv2(concat3)
        # 
        #     concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        #     out_interconv2 = self.inter_conv2(concat2)
        #     flow2 = self.predict_flow2(out_interconv2)
        # 
        # # NOTE: Put aux loss on flow on all levels?
        # # if self.training:
        # #     return flow2,flow3,flow4,flow5,flow6
        # # else:
        # #     return flow2,
        # 
        # x = out_conv6
        # 
        # with torch.set_grad_enabled(self._configs.aux_tasks.fg_mask):
        #     flow = self.upsample1(flow2)
        #     fg_mask2 = self.predict_mask2(out_interconv2)
        #     fg_mask = self.upsampled_mask2_to_0(fg_mask2)
        # 
        # return x, fg_mask, flow

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self.verify_head_layer_specs(self._configs['model']['head_layer_specs']) # Access by key important - needs to be mutable

        if self._configs.model.flownet2_opts.type == 'FlowNetS':
            self.encoder = FlowNetS_wrapper(
                self._configs,
                input_channels = 6,
                batchNorm = False, # Pretrained without batchnorm
            )
            checkpoint = torch.load('/flownet2_models/FlowNet2-S_checkpoint.pth.tar')
            expected_params = set(list(zip(*self.encoder.named_parameters()))[0])
            saved_params = set(checkpoint['state_dict'].keys())
            assert saved_params <= expected_params
            assert all('predict_mask' in pname for pname in expected_params - saved_params)
            self.encoder.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            assert self._configs.model.flownet2_opts.type == 'FlowNetSD'
            self.encoder = FlowNetSD_wrapper(
                self._configs,
                batchNorm = False, # Pretrained without batchnorm
            )
            checkpoint = torch.load('/flownet2_models/FlowNet2-SD_checkpoint.pth.tar')
            expected_params = set(list(zip(*self.encoder.named_parameters()))[0])
            saved_params = set(checkpoint['state_dict'].keys())
            assert saved_params <= expected_params
            assert all('predict_mask' in pname for pname in expected_params - saved_params)
            self.encoder.load_state_dict(checkpoint['state_dict'], strict=False)

        encoder_output_dims = self._check_encoder_output_dims()

        self.heads = nn.ModuleDict({ head_name: Head(self._configs, encoder_output_dims, list(map(AttrDict, head_spec['layers']))) for head_name, head_spec in self._configs.model.head_layer_specs.items() })

    def freeze_encoder(self):
        for param in list(self.E12.parameters()) + list(self.E2.parameters()):
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in list(self.E12.parameters()) + list(self.E2.parameters()):
            param.requires_grad = True

    def _check_encoder_output_dims(self):
        x, fg_mask, flow = self.encoder(
            torch.zeros((1, 3, self._configs.data.crop_dims[0], self._configs.data.crop_dims[1])),
            torch.zeros((1, 3, self._configs.data.crop_dims[0], self._configs.data.crop_dims[1])),
        )
        return x.shape[1:]

    def get_optim_param_groups(self):
        weight_decay = self._configs.model.flownet2_opts.weight_decay if self._configs.model.flownet2_opts.weight_decay is not None else 0.0
        param_groups = []
        param_groups.append({
            'params': [param for name, param in self.named_parameters() if name.endswith('weight') and param.requires_grad],
            'weight_decay': self._configs.model.flownet2_opts.weight_decay,
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
        x, fg_mask, flow = self.encoder(ref_img, query_img)
        x = { head_name: self.heads[head_name](x) for head_name in self.heads }
        # x = self.head(x)

        return {
            'features': x,
            'fg_mask': fg_mask,
            'flow': flow,
        }

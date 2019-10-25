import numpy as np
import torch
from torch import nn

from lib.utils import get_module_parameters, get_device

class ConvBatchReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SiameseUnit(nn.Module):
    """
    Takes a pair of siamese inputs.
    Siamese inputs are fed separately to the same conv operation, resulting in corresponding siamese outputs.
    """
    def __init__(self, configs, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, domain_specific_params=False):
        super().__init__()
        self._configs = configs
        self._domain_specific_params = domain_specific_params
        if self._configs.model.siamese_net:
            if self._domain_specific_params:
                self.conv_real = ConvBatchReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
                self.conv_synth = ConvBatchReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
            else:
                self.conv = ConvBatchReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        else:
            assert not self._domain_specific_params, 'Only siamese net supports the "domain_specific_params" option.'
            self.conv_ref = ConvBatchReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.conv_query = ConvBatchReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        extra_input = x[0]
        x = x[1:]
        y = []
        for i in range(2):
            z = x[i]
            if self._configs.model.siamese_net:
                if self._domain_specific_params:
                    if i == 0:
                        # NOTE: Convolutions applied for all samples in both branches.
                        # Might be slightly more efficient if this is avoided.
                        z = torch.where(
                            extra_input.real_ref.reshape(-1, 1, 1, 1), # Singleton dimensions C,H,W will be broadcasted
                            self.conv_real(z),
                            self.conv_synth(z),
                        )
                    else:
                        z = self.conv_synth(z)
                else:
                    z = self.conv(z)
            else:
                if i == 0:
                    z = self.conv_ref(z)
                else:
                    z = self.conv_query(z)
            y.append(z)
        return y

# model:
#   coord_maps:
#     enabled: False
#     viewing_ray_dependent_coordinates: True
#     preserve_scale: False
class MergeUnit(nn.Module):
    """
    Takes a number of feature maps as input. None elements allowed, but disregarded.
    Concatenates maps, and feeds them through a conv operation.
    """
    def __init__(self, configs, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self._configs = configs
        self.has_prev_merged = in_channels[-1] is not None
        total_in_channels = sum(in_channels) if self.has_prev_merged else sum(in_channels[:-1])
        if self._configs.model.coord_maps.enabled:
            total_in_channels += 2
        self.conv = ConvBatchReLU(total_in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        extra_input = x[0]
        x = x[1:]
        x = [fmap for fmap in x if fmap is not None]
        batch_size, _, ds_height, ds_width = x[0].shape
        if self._configs.model.coord_maps.enabled:
            coord_maps = torch.empty((batch_size, 2, ds_height, ds_width), device=get_device())
            for sample_idx in range(batch_size):
                (x1, y1, x2, y2) = extra_input.crop_box_normalized[sample_idx]
                if self._configs.model.coord_maps.viewing_ray_dependent_coordinates:
                    # Bounding box center in normalized coordinates
                    xc = 0.5 * (x1 + x2)
                    yc = 0.5 * (y1 + y2)
                else:
                    xc = 0.
                    yc = 0.
                if self._configs.model.coord_maps.preserve_scale:
                    # "Preserving the scale" effectively preserves some information which is dependent on absolute depth
                    # Bounding box width & height are in normalized coordinates.
                    cc_width  = x2 - x1
                    cc_height = y2 - y1
                else:
                    cc_width = 2.
                    cc_height = 2.

                # Map width & height from edge-to-edge into distance between centers of edge pixels:
                cc_width  *= (ds_width - 1) / ds_width
                cc_height *= (ds_height - 1) / ds_height

                x_coord_vals = torch.linspace(xc-0.5*cc_width, xc+0.5*cc_width, steps=ds_width, device=get_device())
                y_coord_vals = torch.linspace(yc-0.5*cc_height, yc+0.5*cc_height, steps=ds_height, device=get_device())
                coord_maps[sample_idx, :, :, :] = torch.stack(torch.meshgrid(y_coord_vals, x_coord_vals))
            x.append(coord_maps)
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        return x

class PassThroughLastMap(nn.Module):
    """
    Takes a number of feature maps as input.
    Outputs the last feature map, disregarding the others.
    """
    def __init__(self, ds_factor):
        super().__init__()
        assert ds_factor == 1

    def forward(self, x):
        return x[-1]

class SemiSiameseCNN(nn.Module):
    """
    Siamese network, i.e. one CNN shared for two images, each having an independent data "stream".
    Additionally however, there is a third "merged" data stream, which has access to all levels of the siamese streams.
    The output from the semi-siamese network is the final feature map from the merged stream.
    """
    def __init__(self, configs, cnn_layers):
        super().__init__()
        self._configs = configs
        self.cnn_layers = cnn_layers
        siamese_units_list = []
        merge_units_list = []

        in_channels_siamese = 3
        in_channels_merged = None
        for i, layer_spec in enumerate(self.cnn_layers):
            out_channels = layer_spec.n_out
            kernel_size = layer_spec.kernel_size
            stride = layer_spec.stride
            padding = (layer_spec.kernel_size // 2) * layer_spec.dilation
            if i < len(self.cnn_layers) - 1:
                siamese_units_list.append(SiameseUnit(self._configs, in_channels_siamese, out_channels, kernel_size, stride=stride, padding=padding, domain_specific_params=layer_spec.domain_specific_params))
            if layer_spec.merge:
                # if in_channels_merged is None:
                #     tmp = [in_channels_siamese]*2
                # else:
                #     tmp = [in_channels_siamese]*2 + [in_channels_merged]
                tmp = [in_channels_siamese]*2 + [in_channels_merged]
                merge_units_list.append(MergeUnit(self._configs, tmp, out_channels, kernel_size, stride=stride, padding=padding))
            else:
                merge_units_list.append(PassThroughLastMap(ds_factor=stride))
            in_channels_siamese = out_channels
            if layer_spec.merge:
                in_channels_merged = out_channels

        assert len(merge_units_list) == len(siamese_units_list) + 1

        self.siamese_units = nn.Sequential(*siamese_units_list)
        self.merge_units = nn.Sequential(*merge_units_list)

    def forward(self, x):
        input_maps, extra_input = x

        # Initialize
        s1 = input_maps.ref_img
        s2 = input_maps.query_img
        # mask1 = input_maps.ref_silmask
        # mask2 = input_maps.query_silmask
        mrg = None # Unused

        # Loop through layers
        for i in range(len(self.cnn_layers) - 1):
            s1_next, s2_next = self.siamese_units[i]([extra_input, s1, s2])
            mrg_next = self.merge_units[i]([extra_input, s1, s2, mrg])
            s1, s2, mrg = s1_next, s2_next, mrg_next

        # Final layer
        mrg = self.merge_units[-1]([extra_input, s1, s2, mrg])

        return mrg

class Head(nn.Module):
    def __init__(self, configs, feature_map_dims, head_layers):
        super().__init__()
        self._configs = configs
        self.feature_map_dims = feature_map_dims
        units = []
        in_features = np.prod(self.feature_map_dims)
        for i, layer_spec in enumerate(head_layers):
            out_features = layer_spec.n_out
            dropout_factor = layer_spec.dropout_factor
            relu_flag = layer_spec.relu_flag
            units.append(nn.Linear(in_features, out_features, bias=layer_spec.bias))
            if relu_flag:
                units.append(nn.ReLU(inplace=True))
            if dropout_factor is not None:
                units.append(nn.Dropout(p=dropout_factor))
            in_features = out_features
        self.sequential = nn.Sequential(*units)

    def forward(self, x):
        assert tuple(x.shape[1:]) == tuple(self.feature_map_dims)

        # Global average pooling (channel-wise pooling) vs flatten..?
        x = x.view(x.shape[0], -1) # Flatten

        x = self.sequential(x)

        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self.ds_factor, self.cnn_output_dims = self.calc_downsampling_dimensions(self._configs.model.cnn_layers)
        self.verify_cnn_layer_specs(self._configs.model.cnn_layers)
        self.verify_head_layer_specs(self._configs['model']['head_layers']) # Access by key important - needs to be mutable
        self.semi_siamese_cnn = SemiSiameseCNN(self._configs, self._configs.model.cnn_layers)
        self.head = Head(self._configs, self.cnn_output_dims, self._configs.model.head_layers)

    def get_last_layer_params(self):
        for module in reversed(self.head.sequential):
            w_params, b_params = get_module_parameters(module)
            if len(w_params) > 0 or len(b_params) > 0:
                assert len(w_params) == 1 and len(b_params) == (1 if self._configs.model.head_layers[-1].bias else 0)
                break
        else:
            assert False, "Tried to find last parameterized layer, but found no layer with parameters of interest"
        return w_params, b_params

    def calc_downsampling_dimensions(self, cnn_specs):
        ds_factor = np.prod([layer_spec.stride for layer_spec in cnn_specs])
        cnn_output_dims = [cnn_specs[-1].n_out] + list(np.array(self._configs.data.crop_dims) // ds_factor)
        return ds_factor, cnn_output_dims

    def verify_cnn_layer_specs(self, cnn_specs):
        assert self._configs.data.crop_dims[0] % self.ds_factor == 0
        assert self._configs.data.crop_dims[1] % self.ds_factor == 0
        assert cnn_specs[-1].merge == True

    def verify_head_layer_specs(self, head_specs):
        n_out = sum([self._configs.targets[task_spec['target']]['n_out'] for task_spec in self._configs.tasks.values()])
        if 'n_out' in head_specs[-1]:
            assert head_specs[-1]['n_out'] == n_out
        else:
            head_specs[-1]['n_out'] = n_out
        assert head_specs[-1]['relu_flag'] == False

    def forward(self, x):
        x = self.semi_siamese_cnn(x)
        x = self.head(x)
        return x

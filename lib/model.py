import numpy as np
import torch
from torch import nn

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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = ConvBatchReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        y = []
        for i in range(2):
            z = x[i]
            z = self.conv(z)
            y.append(z)
        return y

class MergeUnit(nn.Module):
    """
    Takes a number of feature maps as input. None elements allowed, but disregarded.
    Concatenates maps, and feeds them through a conv operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.has_prev_merged = in_channels[-1] is not None
        total_in_channels = sum(in_channels) if self.has_prev_merged else sum(in_channels[:-1])
        self.conv = ConvBatchReLU(total_in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        x = [fmap for fmap in x if fmap is not None]
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
    def __init__(self, cnn_layers):
        super().__init__()
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
                siamese_units_list.append(SiameseUnit(in_channels_siamese, out_channels, kernel_size, stride=stride, padding=padding))
            if layer_spec.merge:
                # if in_channels_merged is None:
                #     tmp = [in_channels_siamese]*2
                # else:
                #     tmp = [in_channels_siamese]*2 + [in_channels_merged]
                tmp = [in_channels_siamese]*2 + [in_channels_merged]
                merge_units_list.append(MergeUnit(tmp, out_channels, kernel_size, stride=stride, padding=padding))
            else:
                merge_units_list.append(PassThroughLastMap(ds_factor=stride))
            in_channels_siamese = out_channels
            if layer_spec.merge:
                in_channels_merged = out_channels

        assert len(merge_units_list) == len(siamese_units_list) + 1

        self.siamese_units = nn.Sequential(*siamese_units_list)
        self.merge_units = nn.Sequential(*merge_units_list)

    def forward(self, x):
        # Initialize
        s1, s2 = x
        mrg = None # Unused

        # Loop through layers
        for i in range(len(self.cnn_layers) - 1):
            s1_next, s2_next = self.siamese_units[i]([s1, s2])
            mrg_next = self.merge_units[i]([s1, s2, mrg])
            s1, s2, mrg = s1_next, s2_next, mrg_next

        # Final layer
        mrg = self.merge_units[-1]([s1, s2, mrg])

        return mrg

class Head(nn.Module):
    def __init__(self, feature_map_dims, head_layers):
        super().__init__()
        self.feature_map_dims = feature_map_dims
        units = []
        in_features = np.prod(self.feature_map_dims)
        for i, layer_spec in enumerate(head_layers):
            out_features = layer_spec.n_out
            dropout_factor = layer_spec.dropout_factor
            relu_flag = layer_spec.relu_flag
            units.append(nn.Linear(in_features, out_features))
            if relu_flag:
                units.append(nn.ReLU(inplace=True))
            if dropout_factor is not None:
                units.append(nn.Dropout2d(p=dropout_factor, inplace=True))
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
        self.configs = configs
        self.ds_factor, self.cnn_output_dims = self.calc_downsampling_dimensions(self.configs.model.cnn_layers)
        self.verify_cnn_layer_specs(self.configs.model.cnn_layers)
        self.verify_head_layer_specs(self.configs.model.head_layers)
        self.semi_siamese_cnn = SemiSiameseCNN(self.configs.model.cnn_layers)
        self.head = Head(self.cnn_output_dims, self.configs.model.head_layers)

    def calc_downsampling_dimensions(self, cnn_specs):
        ds_factor = np.prod([layer_spec.stride for layer_spec in cnn_specs])
        cnn_output_dims = [cnn_specs[-1].n_out] + list(np.array(self.configs.data.img_dims) // ds_factor)
        return ds_factor, cnn_output_dims

    def verify_cnn_layer_specs(self, cnn_specs):
        assert self.configs.data.img_dims[0] % self.ds_factor == 0
        assert self.configs.data.img_dims[1] % self.ds_factor == 0
        assert cnn_specs[-1].merge == True

    def verify_head_layer_specs(self, head_specs):
        assert head_specs[-1].n_out == 1
        assert head_specs[-1].relu_flag == False

    def forward(self, x):
        x = self.semi_siamese_cnn(x)
        x = self.head(x)
        return x

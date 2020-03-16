from torch import nn

class Head(nn.Module):
    def __init__(self, configs, feature_map_dims, head_layer_specs):
        super().__init__()
        self._configs = configs
        self.feature_map_dims = feature_map_dims
        units = []
        in_features = np.prod(self.feature_map_dims)
        for i, layer_spec in enumerate(head_layer_specs):
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

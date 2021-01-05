import torch
from torch import nn
import numpy as np
import util


class ImplicitNet(nn.Module):
    """
    Represents a MLP;
    Original code from IGR
    """

    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        d_out=4,
        geometric_init=True,
        radius_init=0.3,
        beta=0.0,
        output_init_gain=2.0,
        num_position_inputs=3,
        sdf_scale=1.0,
        dim_excludes_skip=False,
        combine_layer=1000,
        combine_type="average",
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param geometric_init if true, uses geometric initialization
               (to SDF of sphere)
        :param radius_init if geometric_init, then SDF sphere will have
               this radius
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        :param output_init_gain output layer normal std, only used for
                                output dimension >= 1, when d_out >= 1
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()

        dims = [d_in] + dims + [d_out]
        if dim_excludes_skip:
            for i in range(1, len(dims) - 1):
                if i in skip_in:
                    dims[i] += d_in

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.dims = dims
        self.combine_layer = combine_layer
        self.combine_type = combine_type

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            # if true preform geometric initialization
            if geometric_init:
                if layer == self.num_layers - 2:
                    # Note our geometric init is negated (compared to IDR)
                    # since we are using the opposite SDF convention:
                    # inside is +
                    nn.init.normal_(
                        lin.weight[0],
                        mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]) * sdf_scale,
                        std=0.00001,
                    )
                    nn.init.constant_(lin.bias[0], radius_init)
                    if d_out > 1:
                        # More than SDF output
                        nn.init.normal_(lin.weight[1:], mean=0.0, std=output_init_gain)
                        nn.init.constant_(lin.bias[1:], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                if d_in > num_position_inputs and (layer == 0 or layer in skip_in):
                    # Special handling for input to allow positional encoding
                    nn.init.constant_(lin.weight[:, -d_in + num_position_inputs :], 0.0)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            # Vanilla ReLU
            self.activation = nn.ReLU()

    def forward(self, x, combine_inner_dims=(1,)):
        """
        :param x (..., d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        x_init = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer == self.combine_layer:
                x = util.combine_interleaved(x, combine_inner_dims, self.combine_type)
                x_init = util.combine_interleaved(
                    x_init, combine_inner_dims, self.combine_type
                )

            if layer < self.combine_layer and layer in self.skip_in:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            conf.get_list("dims"),
            skip_in=conf.get_list("skip_in"),
            beta=conf.get_float("beta", 0.0),
            dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            **kwargs
        )

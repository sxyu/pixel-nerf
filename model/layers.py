from torch import nn
import util
import torch.autograd.profiler as profiler


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetBlock2D(nn.Module):
    """
    Very basic ResBlock for 2D.
    """

    def __init__(
        self,
        dim,
        padding_type="reflect",
        last=False,
        use_bias=False,
        norm_layer=util.get_norm_layer("group"),
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.conv1 = util.make_conv_2d(
            dim,
            dim,
            padding_type,
            norm_layer=norm_layer,
            use_bias=use_bias,
            activation=activation,
        )
        self.conv2 = util.make_conv_2d(
            dim, dim, padding_type, None if last else norm_layer, use_bias=use_bias
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class ResnetBlock3D(nn.Module):
    """
    Very basic ResBlock for 3D.
    """

    def __init__(
        self,
        dim,
        padding_type="replicate",
        last=False,
        use_bias=False,
        norm_layer=util.get_norm_layer("group"),
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.conv1 = util.make_conv_3d(
            dim,
            dim,
            padding_type,
            norm_layer=norm_layer,
            use_bias=use_bias,
            activation=activation,
        )
        self.conv2 = util.make_conv_3d(
            dim, dim, padding_type, None if last else norm_layer, use_bias=use_bias
        )
        #  , zero_init=True)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x

import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from model.layers import ResnetBlock2D


class Resnet(nn.Module):
    """
    Basic custom convolutional encoder with resblocks
    """

    def __init__(
        self,
        dim_in=3,
        dims=[128, 128, 128, 256],
        norm_layer=util.get_norm_layer("group"),
        padding_type="reflect",
        use_skip_conn=True,
        use_leaky_relu=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.use_skip_conn = use_skip_conn
        self.padding_type = padding_type

        first_layer_chnls = dims[0]
        n_blocks = len(dims)
        self.n_blocks = n_blocks
        self.dims = dims

        self.conv0 = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=1, bias=False),
            norm_layer(first_layer_chnls),
            self.activation,
        )

        for i in range(n_blocks):
            if i > 0:
                convi = nn.Sequential(
                    nn.Conv2d(
                        dims[i - 1],
                        dims[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    norm_layer(dims[i]),
                    self.activation,
                )
                setattr(self, "conv" + str(i), convi)

            blki = ResnetBlock2D(
                dims[i],
                norm_layer=self.norm_layer,
                padding_type=padding_type,
                activation=self.activation,
            )
            setattr(self, "resblk" + str(i), blki)

        for i in reversed(range(self.n_blocks)):
            in_chnls = dims[i]
            if i < n_blocks - 1 and use_skip_conn:
                in_chnls *= 2
            if i > 0:
                deconvi = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_chnls,
                        dims[i - 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    norm_layer(dims[i - 1]),
                    self.activation,
                )
                setattr(self, "deconv" + str(i), deconvi)
            if i < n_blocks - 1:
                blkupi = ResnetBlock2D(
                    in_chnls,
                    norm_layer=self.norm_layer,
                    padding_type=padding_type,
                    activation=self.activation,
                )
                setattr(self, "resblkup" + str(i), blkupi)

    def forward(self, x):
        # *** Image network ***
        x = self.conv0(x)
        inter = []
        for i in range(self.n_blocks):
            if i > 0:
                x = getattr(self, "conv" + str(i))(x)
            x = getattr(self, "resblk" + str(i))(x)
            inter.append(x)

        for i in reversed(range(self.n_blocks)):
            if i < self.n_blocks - 1:
                if self.use_skip_conn:
                    x = torch.cat((x, inter[i]), dim=1)
                x = getattr(self, "resblkup" + str(i))(x)
            if i > 0:
                x = getattr(self, "deconv" + str(i))(x)
        return x

    @classmethod
    def from_conf(cls, conf, dim_in=3):
        return cls(
            dim_in,
            dims=conf.get_list("dims", [128, 128, 128, 256]),
            norm_layer=util.get_norm_layer(conf.get_string("norm_type", "group")),
            padding_type=conf.get_string("padding_type", "reflect"),
            use_leaky_relu=conf.get_bool("use_leaky_relu", True),
            use_skip_conn=conf.get_bool("use_skip_conn", True),
        )


class ConvEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    """

    def __init__(
        self,
        dim_in=3,
        norm_layer=util.get_norm_layer("group"),
        padding_type="reflect",
        use_leaky_relu=True,
        use_skip_conn=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.padding_type = padding_type
        self.use_skip_conn = use_skip_conn

        # TODO: make these configurable
        first_layer_chnls = 64
        mid_layer_chnls = 128
        last_layer_chnls = 128
        n_down_layers = 3
        self.n_down_layers = n_down_layers

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=7, stride=2, bias=False),
            norm_layer(first_layer_chnls),
            self.activation,
        )

        chnls = first_layer_chnls
        for i in range(0, n_down_layers):
            conv = nn.Sequential(
                nn.Conv2d(chnls, 2 * chnls, kernel_size=3, stride=2, bias=False),
                norm_layer(2 * chnls),
                self.activation,
            )
            setattr(self, "conv" + str(i), conv)

            deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    4 * chnls, chnls, kernel_size=3, stride=2, bias=False
                ),
                norm_layer(chnls),
                self.activation,
            )
            setattr(self, "deconv" + str(i), deconv)
            chnls *= 2

        self.conv_mid = nn.Sequential(
            nn.Conv2d(chnls, mid_layer_chnls, kernel_size=4, stride=4, bias=False),
            norm_layer(mid_layer_chnls),
            self.activation,
        )

        self.deconv_last = nn.ConvTranspose2d(
            first_layer_chnls, last_layer_chnls, kernel_size=3, stride=2, bias=True
        )

        self.dims = [last_layer_chnls]

    def forward(self, x):
        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        x = self.conv_in(x)

        inters = []
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = util.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x)
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        for i in reversed(range(0, self.n_down_layers)):
            if self.use_skip_conn:
                x = torch.cat((x, inters[i]), dim=1)
            deconv_i = getattr(self, "deconv" + str(i))
            x = deconv_i(x)
            x = util.same_unpad_deconv2d(x, layer=deconv_i)
        x = self.deconv_last(x)
        x = util.same_unpad_deconv2d(x, layer=self.deconv_last)
        return x

"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import util
from model.layers import ResnetBlock2D, ResnetBlock3D
from model.resnet import Resnet, ConvEncoder
import torch.autograd.profiler as profiler


class PIFuEncoder(nn.Module):
    """
    2D (PIFu/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        custom_type="resnet",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        low_memory=False,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.resnet.Resnet or model.resnet.ConvEncoder is used,
        depending on setting of custom_type (default former)
        or resnet*, in which case a model
        from torchvision is used
        e.g. resnet34 | resnet18
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param custom_type if backbone = custom, resnet | simple for custom encoder type
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much
        :param norm_type norm type to applied; pretrained model uses batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.custom_type = custom_type
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            if custom_type == "resnet":
                print("Using custom Resnet encoder")
                self.model = Resnet(3)
                self.latent_size = self.model.dims[0] * 2
            elif custom_type == "simple":
                print("Using simple convolutional encoder")
                self.model = ConvEncoder(3)
                self.latent_size = self.model.dims[-1]
            else:
                raise NotImplementedError(
                    "Unsupported encoder custom_type", custom_type
                )
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.low_memory = low_memory
        print("LOW MEMORY ON?", self.low_memory)
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer("latent_scaling", torch.empty(2, dtype=torch.float32),
                persistent=False)
        # self.latent (B, L, H, W)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get PIFu features at image points
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if self.low_memory:
                return self.index_low_memory(uv, image_size)

            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def index_low_memory(self, uv, image_size=()):
        # concatenate latents (B, C, N) along first dimension
        if uv.shape[0] == 1 and self.latents[0].shape[0] > 1:
            uv = uv.expand(self.latents[0].shape[0], -1, -1)
        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        return torch.cat(
            [self._index_single(i, uv, image_size) for i in range(len(self.latents))],
            dim=1,
        )

    def _index_single(self, latent_idx, uv, image_size):
        latent = self.latents[latent_idx]

        if len(image_size) > 0:
            if len(image_size) == 1:
                image_size = (image_size, image_size)
            latent_sz = latent.shape[-2:]
            width_scale = latent_sz[1] / image_size[0] * 2.0 / (latent_sz[1] - 1)
            height_scale = latent_sz[0] / image_size[1] * 2.0 / (latent_sz[0] - 1)
            scale = torch.tensor((width_scale, height_scale), device=uv.device)
            uv = uv * scale - 1.0

        samples = F.grid_sample(
            latent,
            uv,
            align_corners=True,
            mode=self.index_interp,
            padding_mode=self.index_padding,
        )
        return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            if not self.low_memory:
                align_corners = None if self.index_interp == "nearest " else True
                latent_sz = latents[0].shape[-2:]
                for i in range(len(latents)):
                    latents[i] = F.interpolate(
                        latents[i],
                        latent_sz,
                        mode=self.upsample_interp,
                        align_corners=align_corners,
                    )
                self.latent = torch.cat(latents, dim=1)
        if not self.low_memory:
            self.latent_scaling[0] = self.latent.shape[-1]
            self.latent_scaling[1] = self.latent.shape[-2]
            self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            custom_type=conf.get_string("custom_type", "resnet"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    0D (global) image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )

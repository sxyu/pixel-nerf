from .encoder import SpatialEncoder, ImageEncoder
from .resnetfc import ResnetFC


def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get_string("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net

from .models import PixelNeRFNet


def make_model(conf, *args, **kwargs):
    model_type = conf.get_string("type", "pixelnerf")  # single
    # Placeholder
    if model_type == "single":
        net = PixelNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net

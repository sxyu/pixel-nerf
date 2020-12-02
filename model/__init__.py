from .models import PIFuNeRFNet


def make_model(conf, *args, **kwargs):
    model_type = conf.get_string("type", "single")  # single | multi
    if model_type == "single":
        net = PIFuNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net

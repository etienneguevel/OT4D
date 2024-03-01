import yaml
from models import Resnet18, ViT


def load_model(model_name, ckpt_name, num_classes):
    """
    function loading one of the implemented models
    -
    """
    if model_name == "resnet18":
        model = Resnet18.load_from_checkpoint(
            ckpt_name,
            num_classes=num_classes,
        )

    elif model_name == "ViT":
        model = ViT.load_from_checkpoint(
            ckpt_name,
            model_name="vit_base_patch16_224.orig_in21k",
            num_classes=num_classes,
        )

    else:
        raise ValueError("Sorry, the name of the model used is unvalid")

    return model


def initialize_model(model_name, **kwargs):
    if model_name == "resnet18":
        model = Resnet18(**kwargs)

    elif model_name == "ViT":
        model_name = "vit_base_patch16_224.orig_in21k"
        model = ViT(model_name=model_name, **kwargs)

    else:
        return ValueError("Sorry, the name of the model is unvalid")

    return model


def add_to_yaml(yaml_file, key, value):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
        data[key] = value

    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

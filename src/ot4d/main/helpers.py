import yaml
from .models import Resnet18, ViT


def load_model(model_name: str, ckpt_name: str, num_classes: int):
    """
    Model loader from a checkpoint file
    params:
    - model_name: name of the model class to load, from `resnet18` and `ViT`
    - ckpt_name: path of the checkpoint to load the model from
    - num_classes: number of classes to classify from

    outs:
    - model class of the `model_name` specified with the weight from the `ckpt_name` path
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


def initialize_model(model_name: str, **kwargs):
    """
    Function to initialize a model from model_classes specified
    args:
    - model_name: name of the model class to load, from `resnet18` and `ViT`

    outs:
    - model class of the `model_name` specified
    """
    if model_name == "resnet18":
        model = Resnet18(**kwargs)

    elif model_name == "ViT":
        model_name = "vit_base_patch16_224.orig_in21k"
        model = ViT(model_name=model_name, **kwargs)

    else:
        return ValueError("Sorry, the name of the model is unvalid")

    return model


def add_to_yaml(yaml_file: str, key, value):
    """
    Function to add content to a yaml file
    args:
    - yaml_file: path to the yaml file to modify
    - key: key to add
    - value: value to add
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
        data[key] = value

    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

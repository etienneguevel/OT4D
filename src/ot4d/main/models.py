import lightning as L
import timm
import torch
from torch import nn, optim
from torchvision import models, transforms


class ViT(L.LightningModule):
    """
    Vision Transformer built from timm library (Huggingface library for vision models),
    and made to be trained by the Lightning interface.
    The model is made for image classification.
    The optimizer used is SGD, and a learning rate scheduler is used.
    args:
    - model_name: name of the model that we will query from timm
    - num_classes: number of classes
    - lr: learning rate for the training
    - momentum: momentum used for SGD
    - step_size: number of steps before decreasing the learning rate
    - gamma: parameter to decrease the learning rate

    outs:
    - ViT object

    methods:
    - training_step: method to specify to lightning how to use the model during training time
    - validation_step: method to specify to lightning how to use the model during the validation time
    - configure_optimizers: method to specify to lightning how to configure the optimizer
    - processor: return the function used by the model to preprocess the image
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 1e-3,
        momentum: float = 0.9,
        step_size: int = 10,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        self.lr = float(lr)
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        _, preds = torch.max(outputs, 1)

        acc = torch.sum(y == preds) / len(y)
        self.log("train_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        self.log("val_loss", loss, prog_bar=True)

        _, preds = torch.max(outputs, 1)

        acc = torch.sum(y == preds) / len(y)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return [optimizer], [lr_scheduler]

    def processor(self):
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        processor = timm.data.create_transform(**data_cfg)

        return processor


class Resnet18(L.LightningModule):
    """
    Resnet18 model built from the torchvision library, and made to be trained
    by the Lighting interface.
    The model is made for image classification.
    args:
    - model_name: name of the model that we will query from timm
    - num_classes: number of classes
    - lr: learning rate for the training
    - momentum: momentum used for SGD
    - step_size: number of steps before decreasing the learning rate
    - gamma: parameter to decrease the learning rate

    outs:
    - Resnet18 object

    methods:
    - training_step: method to specify to lightning how to use the model during training time
    - validation_step: method to specify to lightning how to use the model during the validation time
    - configure_optimizers: method to specify to lightning how to configure the optimizer
    - processor: return the function used by the model to preprocess the image

    """

    def __init__(self, num_classes, lr=1e-3, momentum=0.9, step_size=10, gamma=0.1):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.lr = float(lr)
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        _, preds = torch.max(outputs, 1)

        acc = torch.sum(y == preds) / len(y)
        self.log("train_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        self.log("val_loss", loss, prog_bar=True)

        _, preds = torch.max(outputs, 1)

        acc = torch.sum(y == preds) / len(y)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return [optimizer], [lr_scheduler]

    def processor(self):
        processor = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return processor

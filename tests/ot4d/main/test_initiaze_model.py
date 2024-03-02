import skimage
import numpy as np
import torch
from ot4d.main.helpers import initialize_model
from PIL import Image

image = skimage.data.astronaut()
image = Image.fromarray(np.uint8(image)).convert("RGB")


def test_initialize():
    for m in ["resnet18", "ViT"]:
        model = initialize_model(model_name=m, num_classes=2)
        processor = model.processor()
        image_processed = processor(image)

        image_processed = torch.from_numpy(np.expand_dims(image_processed, axis=0))
        logits = model.model(image_processed)

        assert logits.shape[1] == 2

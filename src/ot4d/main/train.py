import os
import sys

import lightning as L
import torch
import yaml
from ot4d.main.helpers import add_to_yaml, initialize_model
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchvision import datasets

from dvclive.lightning import DVCLiveLogger


def make_dataloaders(data_dir, processor, batch_size):
    """
    Function to create torch dataloaders in order to load pictures in a directory and return them in a dataloaders.
    The pictures needs to be organised into 3 folders : train, test, validation.
    In each folder the different pictures concerning a same class are grouped together in a folder named as the class.

    :param data_dir: path to the directory containing the pictures
    :return: torch dataloaders for the train, validation and test folders, and the list of the class names inside the folders
    """

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), processor)
        for x in ["train", "validation", "test"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "validation", "test"]
    }

    return dataloaders


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)

    # Import the name of the directory
    data_dir = sys.argv[1]

    # Load the parameters specified in the params file
    with open("params.yaml") as file:
        f = yaml.safe_load(file)
        params = f["train"]
        model_name = f["global"]["model_name"]
        path = f["global"]["savepath"]

    # Check the devices available
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Count the number of classes (ie number of files in any datasplit directory)
    num_classes = len(os.listdir(os.path.join(data_dir, "train")))

    # Initialize the model with the specified parameters
    model = initialize_model(
        model_name=model_name,
        num_classes=num_classes,
        lr=params["learning_rate"],
        momentum=params["momentum"],
        step_size=params["step_size"],
        gamma=params["gamma"],
    )

    # Make the dataloaders with the model preprocessing function
    dataloaders = make_dataloaders(
        data_dir=data_dir, processor=model.processor(), batch_size=params["batch_size"]
    )

    # Specify the savepath where the best model will be saved
    savepath = os.path.join(path, model_name)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # Make a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=savepath,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )

    # Make an earlystopping callback
    early_stopping = EarlyStopping("val_loss")

    # Instantiate the DVC logger
    dvclive_logger = DVCLiveLogger()

    # Define the trainer with the callbacks / logger / specified parameters
    trainer = L.Trainer(
        max_epochs=params["num_epochs"],
        accelerator=device,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        logger=dvclive_logger,
    )

    # Fit the model
    trainer.fit(model, dataloaders["train"], dataloaders["validation"])

    # Get the name of the best model trained, and write it in the yaml file for the evaluation phase
    best_model = checkpoint_callback.best_model_path
    add_to_yaml("params.yaml", "evaluate", {"ckpt_name": best_model})


if __name__ == "__main__":
    main()

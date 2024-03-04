import os
import shutil
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image


def find_files(path_dir: str):
    """
    help function to find the files in a directory
    args:
    - path_dir: name of the directory to inspect

    outs:
    - files: dictionnary containing as a key the path of the files found and as a value their name in their directory
    """
    files = {}

    for dirpath, dirnames, filenames in os.walk(path_dir):
        temp = {os.path.join(dirpath, f): f for f in filenames}
        files.update(temp)

    return files


def move_data(
    folder_path: str,
    out_path: str,
    validation_split: float = 0.2,
    test_split: float = 0.1,
):
    """
    Function to copy and rearrange the files in folder_path in a format train / validation / test in a
    directory out_path.
    The percentage of files in each folder is determined by `validation_split` and `test_split`.
    The pre-conceived idea between this function is that the data inside `folder_path` is arranged inside
    folders named after the class of the pictures inside them.
    These folders will be replicated inside each of the folders train / validation / test inside out_path.

    args:
    - folder_path: path of the folder containing the original data
    - out_path: path where the new sorted dataset will be copied
    - validation_split: percentage of the data to move in the validation folder
    - test_split: percentage of the date to move in the test folder
    """
    print("Starting")
    val_test_ratio = test_split / (validation_split + test_split)
    states = [fn for fn in os.listdir(folder_path)]

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.mkdir(out_path)
    for s in ["train", "validation", "test"]:
        os.mkdir(os.path.join(out_path, s))

    for i, j in zip(states, tqdm(range(len(states)))):
        imgs = find_files(os.path.join(folder_path, i))
        train, temp = train_test_split(
            list(imgs.keys()), test_size=validation_split + test_split
        )
        validation, test = train_test_split(temp, test_size=val_test_ratio)

        for k, f in zip(["train", "validation", "test"], [train, validation, test]):
            path_f = os.path.join(out_path, k, i)

            if not os.path.isdir(path_f):
                os.mkdir(path_f)

            for in_path in f:
                try:
                    im = Image.open(in_path)
                    im = im.convert("RGB")
                    shutil.copyfile(in_path, os.path.join(path_f, imgs[in_path]))

                except OSError:
                    print(
                        f"Image at path {in_path} could not be opened and has not been copied."
                    )

    print("Files have been moved")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)

    folder_path = sys.argv[1]
    out_path = sys.argv[2]
    move_data(folder_path=folder_path, out_path=out_path)


if __name__ == "__main__":
    main()

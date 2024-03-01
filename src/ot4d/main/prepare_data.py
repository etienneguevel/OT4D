import os
import shutil
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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


def move_data(folder_path, out_path, validation_split=0.2, test_split=0.1):
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
                shutil.copyfile(in_path, os.path.join(path_f, imgs[in_path]))

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

import csv
import logging
import os
import os.path
import tarfile
from typing import Callable
from urllib.parse import urlparse

from PIL import Image
from torch.utils.data import Dataset

from frarch.utils.data import download_url

logger = logging.getLogger(__name__)

urls = {
    "images": "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar",
    "train_file": "http://web.mit.edu/torralba/www/TrainImages.txt",
    "test_file": "http://web.mit.edu/torralba/www/TestImages.txt",
}


def download_mit(root):
    """Download the data."""
    tmpdir = os.path.join(root, "tmp")
    path_images = os.path.join(root, "Images")

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    # create directory
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls["images"])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls["images"], cached_file))
            download_url(urls["images"], cached_file)

        # extract file
        print(f"[dataset] Extracting tar file {cached_file} to {root}")
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print("[dataset] Done!")

    # download train file
    parts = urlparse(urls["train_file"])
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(root, filename)

    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls["train_file"], cached_file))
        download_url(urls["train_file"], cached_file)

    # download test file
    parts = urlparse(urls["test_file"])
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(root, filename)

    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls["test_file"], cached_file))
        download_url(urls["test_file"], cached_file)


def find_classes(dir):
    fname = os.path.join(dir, "TrainImages.txt")
    # read the content of the file
    with open(fname) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    # find the list of classes
    classes = {}
    for x in content:
        classes[x.split("/")[0]] = 0

    # assign a label for each class
    index = 0
    for key in sorted(classes):
        classes[key] = index
        index += 1

    return classes


def make_dataset(dir, classes, set):
    images = []

    if set == "train":
        fname = os.path.join(dir, "TrainImages.txt")
    elif set == "test":
        fname = os.path.join(dir, "TestImages.txt")
    else:
        raise ValueError("set must be train or test")

    # read the content of the file
    with open(fname) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    for x in content:
        path = x
        label = classes[x.split("/")[0]]
        item = (path, label)
        images.append(item)

    return images


def write_csv_file(dir, images, set):
    csv_file = os.path.join(dir, set + ".csv")
    if not os.path.exists(csv_file):

        # write a csv file
        print("[dataset] write file %s" % csv_file)
        with open(csv_file, "w") as csvfile:
            fieldnames = ["name", "label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for x in images:
                writer.writerow({"name": x[0], "label": x[1]})

        csvfile.close()


class Mit67(Dataset):
    def __init__(
        self,
        root,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = True,
        toy: bool = False,
    ):
        self.root = root
        self.set = "train" if train else "test"
        self.transform = transform
        self.target_transform = target_transform
        self.path_images = os.path.join(self.root, "Images")

        if download:
            download_mit(self.root)

        try:
            self.classes = find_classes(self.root)
            self.images = make_dataset(self.root, self.classes, self.set)
            if toy:
                self.images = self.images[:100]
        except OSError as e:
            logger.error(
                f"Files for MIT67 dataset not found. Download them manually in {root}"
                " dir or set the download to True in the Mit67 class to download the"
                " data automatically."
            )
            raise e

        print(
            "[dataset] MIT67 set=%s  number of classes=%d  number of images=%d"
            % (self.set, len(self.classes), len(self.images))
        )

        write_csv_file(self.root, self.images, self.set)

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

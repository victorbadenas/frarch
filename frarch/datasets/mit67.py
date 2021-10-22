import csv
import logging
import os
import os.path
import tarfile
from pathlib import Path
from typing import Callable, Union
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from frarch.utils.data import download_url

logger = logging.getLogger(__name__)

urls = {
    "images": "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
}


class Mit67(Dataset):
    def __init__(
        self,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = True,
    ):
        self.root = Path("~/.cache/frarch/datasets/").expanduser()
        self.set = "train" if train else "test"
        self.transform = transform
        self.target_transform = target_transform

        if download and not self.detect_dataset():
            self.download_mit_dataset()
        elif not download and not self.detect_dataset():
            raise ValueError()

    def detect_dataset(self):
        if not self.root.exists():
            return False
        else:
            num_images = len(list(self.root.glob("**/*.jpeg")))
            return num_images > 0

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path)).convert("RGB")
        img = torch.Tensor(np.array(img)).permute((2, 0, 1)) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

    def download_mit_dataset(self):
        self.root.mkdir(parents=True, exist_ok=True)

        # download train/val images/annotations
        parts = urlparse(urls["images"])
        filename = os.path.basename(parts.path)
        cached_file = self.root / filename

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls["images"], cached_file))
            download_url(urls["images"], cached_file)

        # extract file
        # print(f"[dataset] Extracting tar file {cached_file} to {root}")
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        # os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print("[dataset] Done!")

        # download train file
        parts = urlparse(urls["train_file"])
        filename = os.path.basename(parts.path)
        # cached_file = os.path.join(root, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls["train_file"], cached_file))
            download_url(urls["train_file"], cached_file)

        # download test file
        parts = urlparse(urls["test_file"])
        filename = os.path.basename(parts.path)
        # cached_file = os.path.join(root, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls["test_file"], cached_file))
            download_url(urls["test_file"], cached_file)

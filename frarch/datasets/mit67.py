import csv
import json
import logging
import os
import os.path
import random
import tarfile
from collections import Counter
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
        root: Union[str, Path] = "~/.cache/frarch/datasets/",
        full: bool = False,
    ):
        self.root = Path(root).expanduser()
        self.set = "train" if train else "test"
        self.transform = transform
        self.target_transform = target_transform
        self.full = full

        self.train_lst_path = self.root / "train.lst"
        self.valid_lst_path = self.root / "valid.lst"
        self.mapper_path = self.root / "class_map.json"

        if download and not self._detect_dataset():
            self.download_mit_dataset()
        if not self._detect_dataset():
            raise ValueError(
                f"download flag not set and dataset not present in {self.root}"
            )

        self._build_and_load_data_files()

        print(
            f"Loaded {self.set} Split: {len(self.images)} instances"
            f" in {len(self.classes)} classes"
        )

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(path).convert("RGB")
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
        print(f"[dataset] Extracting tar file {cached_file} to {self.root}")
        tar = tarfile.open(cached_file, "r")
        tar.extractall(self.root)
        tar.close()
        print("[dataset] Done!")

    def _get_file_paths(self):
        return list(self.root.glob("Images/*/*.jpg"))

    def _detect_dataset(self):
        if not self.root.exists():
            return False
        else:
            num_images = len(self._get_file_paths())
            return num_images > 0

    def _build_and_load_data_files(self):
        all_paths = self._get_file_paths()
        self._load_class_map(all_paths)
        self._load_train_test_files(all_paths)

    def _load_class_map(self, all_paths):
        if not self.mapper_path.exists():
            self._build_class_mapper(all_paths)
        with self.mapper_path.open("r") as f:
            self.classes = json.load(f)

    def _build_class_mapper(self, all_paths):
        classes_set = set(map(lambda path: path.parts[-2], all_paths))
        print(f"found {len(classes_set)} classes.")
        class_mapper = dict(zip(classes_set, range(len(classes_set))))
        print(f"class mapper built: {class_mapper}")
        with self.mapper_path.open("w") as f:
            json.dump(class_mapper, f)

    def _load_train_test_files(self, all_paths):
        if not self.train_lst_path.exists() and not self.valid_lst_path.exists():
            self._build_train_test_files(all_paths)
        self._load_set(self.set)

    def _build_train_test_files(self, all_paths):
        classes_list = list(map(lambda path: path.parts[-2], all_paths))
        instance_counter = Counter(classes_list)

        train_instances, valid_instances = [], []
        for class_name, count in instance_counter.items():
            class_instances = list(
                filter(lambda x: x.parts[-2] == class_name, all_paths)
            )
            random.shuffle(class_instances)
            if self.full:
                valid_count = max(1, int(count / 10))
                class_valid_instances = class_instances[:valid_count]
                class_train_instances = class_instances[valid_count:]
            else:
                class_valid_instances = class_instances[:20]
                class_train_instances = class_instances[20:100]

            valid_instances.extend(
                [
                    (instance, self.classes[class_name])
                    for instance in class_valid_instances
                ]
            )
            train_instances.extend(
                [
                    (instance, self.classes[class_name])
                    for instance in class_train_instances
                ]
            )

        print(
            f"Built Train Split: {len(train_instances)} instances"
            f" in {len(self.classes)} classes"
        )
        print(
            f"Built Valid Split: {len(valid_instances)} instances"
            f" in {len(self.classes)} classes"
        )

        with self.train_lst_path.open("w") as f:
            for line in train_instances:
                f.write(",".join(map(str, line)) + "\n")
        with self.valid_lst_path.open("w") as f:
            for line in valid_instances:
                f.write(",".join(map(str, line)) + "\n")

    def _load_set(self, set):
        path = self.train_lst_path if set == "train" else self.valid_lst_path
        with path.open("r") as f:
            self.images = []
            for line in f:
                path, label = line.strip().split(",")
                self.images.append((path, int(label)))

import json
import logging
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Callable
from typing import List
from typing import Union
from urllib.parse import urlparse

import torch
from PIL import Image
from torch.utils.data import Dataset

from frarch.utils.data import download_url
from frarch.utils.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)

urls = {
    "images": "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
}


class Mit67(Dataset):
    """Mit 67 dataset object.

    Data loader for the Mit 67 dataset for indoor scene recognition. The dataset can
    be obtained from
    http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar.

    Args:
        train (bool): True for loading the train subset and False for valid. Defaults
            to True.
        transform (Callable): a callable object that takes an `PIL.Image` object and
            returns a modified `PIL.Image` object. Defaults to None, which won't apply
            any transformation.
        target_transform (Callable): a callable object that the label data and returns
            modified label data. Defaults to None, which won't apply any transformation.
        download (bool): True for downloading and storing the dataset data in the `root`
            directory if it's not present. Defaults to True.
        root (Union[str, Path]): root directory for the dataset.
            Defaults to `~/.cache/frarch/datasets/mit67/`.

    References:
        - http://web.mit.edu/torralba/www/indoor.html

    Examples:
        Simple usage of the dataset class::

            from frarch.datasets import Mit67
            from frarch.utils.data import create_dataloader
            from torchvision.transforms import ToTensor

            dataset = Mit67(True, ToTensor, None, True, "./data/")
            dataloader = create_dataloader(dataset)
            for batch_idx, (batch, labels) in enumerate(dataloader):
                # process batch
    """

    def __init__(
        self,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = True,
        root: Union[str, Path] = "~/.cache/frarch/datasets/mit67/",
    ) -> None:
        self.root: Path = Path(root).expanduser()
        self.set = "train" if train else "test"
        self.transform = transform
        self.target_transform = target_transform

        self.train_lst_path = self.root / "train.lst"
        self.valid_lst_path = self.root / "valid.lst"
        self.mapper_path = self.root / "class_map.json"

        if download and not self._detect_dataset():
            self._download_mit_dataset()
        if not self._detect_dataset():
            raise DatasetNotFoundError(
                f"download flag not set and dataset not present in {self.root}"
            )

        self._build_and_load_data_files()

        logger.info(
            f"Loaded {self.set} Split: {len(self.images)} instances"
            f" in {len(self.classes)} classes"
        )

    def __getitem__(self, index: int) -> Union[torch.Tensor, int]:
        path, target = self.images[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.images)

    def get_number_classes(self) -> int:
        """Get number of target labels.

        Returns:
            int: number of target labels.
        """
        return len(self.classes)

    def _download_mit_dataset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

        # download train/val images/annotations
        parts = urlparse(urls["images"])
        filename = Path(parts.path).name
        cached_file = self.root / filename

        if not cached_file.exists():
            logger.info('Downloading: "{}" to {}\n'.format(urls["images"], cached_file))
            download_url(urls["images"], cached_file)

        # extract file
        logger.info(f"[dataset] Extracting tar file {cached_file} to {self.root}")
        tar = tarfile.open(cached_file, "r")
        tar.extractall(self.root)
        tar.close()
        logger.info("[dataset] Done!")
        cached_file.unlink()

    def _get_file_paths(self) -> List[Path]:
        return list(self.root.glob("Images/*/*.jpg"))

    def _detect_dataset(self) -> bool:
        if not self.root.exists():
            return False
        else:
            num_images = len(self._get_file_paths())
            return num_images > 0

    def _build_and_load_data_files(self) -> None:
        all_paths = self._get_file_paths()
        self._load_class_map(all_paths)
        self._load_train_test_files(all_paths)

    def _load_class_map(self, all_paths: List[Path]) -> None:
        if not self.mapper_path.exists():
            self._build_class_mapper(all_paths)
        with self.mapper_path.open("r") as f:
            self.classes = json.load(f)

    def _build_class_mapper(self, all_paths: List[Path]) -> None:
        classes_set = {path.parts[-2] for path in all_paths}
        logger.info(f"found {len(classes_set)} classes.")
        class_mapper = dict(zip(classes_set, range(len(classes_set))))
        logger.info(f"class mapper built: {class_mapper}")
        with self.mapper_path.open("w") as f:
            json.dump(class_mapper, f)

    def _load_train_test_files(self, all_paths: List[Path]) -> None:
        if not self.train_lst_path.exists() and not self.valid_lst_path.exists():
            self._build_train_test_files(all_paths)
        self._load_set(self.set)

    def _build_train_test_files(self, all_paths: List[Path]) -> None:
        classes_list = [path.parts[-2] for path in all_paths]
        instance_counter = Counter(classes_list)

        train_instances, valid_instances = [], []
        for class_name, count in instance_counter.items():
            class_instances = list(
                filter(lambda x, name=class_name: x.parts[-2] == name, all_paths)
            )
            random.shuffle(class_instances)
            valid_count = max(1, int(count / 10))
            class_valid_instances = class_instances[:valid_count]
            class_train_instances = class_instances[valid_count:]

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

        logger.info(
            f"Built Train Split: {len(train_instances)} instances"
            f" in {len(self.classes)} classes"
        )
        logger.info(
            f"Built Valid Split: {len(valid_instances)} instances"
            f" in {len(self.classes)} classes"
        )

        with self.train_lst_path.open("w") as f:
            for line in train_instances:
                f.write(",".join(map(str, line)) + "\n")
        with self.valid_lst_path.open("w") as f:
            for line in valid_instances:
                f.write(",".join(map(str, line)) + "\n")

    def _load_set(self, set: str) -> None:
        path = self.train_lst_path if set == "train" else self.valid_lst_path
        with path.open("r") as f:
            self.images = []
            for line in f:
                path, label = line.strip().split(",")
                self.images.append((path, int(label)))

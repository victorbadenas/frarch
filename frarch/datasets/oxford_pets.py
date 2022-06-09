import logging
import tarfile
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
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}


class OxfordPets(Dataset):
    """Oxford Pets dataset object.

    Data loader for the Oxford Pets dataset for pet recognition. The dataset can be
    obtained from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz and
    their corresponding labels in
    https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.

    Args:
        subset (str): "train" or "valid". Subset to load. Defaults to "train".
        transform (Callable): a callable object that takes an `PIL.Image` object and
            returns a modified `PIL.Image` object. Defaults to None, which won't apply
            any transformation.
        target_transform (Callable): a callable object that the label data and returns
            modified label data. Defaults to None, which won't apply any transformation.
        download (bool): True for downloading and storing the dataset data in the `root`
            directory if it's not present. Defaults to True.
        root (Union[str, Path]): root directory for the dataset.
            Defaults to `~/.cache/frarch/datasets/oxford_pets/`.

    References:
        - https://www.robots.ox.ac.uk/~vgg/data/pets/

    Examples:
        Simple usage of the dataset class::

            from frarch.datasets import Mit67
            from frarch.utils.data import create_dataloader
            from torchvision.transforms import ToTensor

            dataset = OxfordPets( "train", ToTensor, None, True, "./data/")
            dataloader = create_dataloader(dataset)
            for batch_idx, (batch, labels) in enumerate(dataloader):
                # process batch
    """

    def __init__(
        self,
        subset: str = "train",
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = True,
        root: Union[str, Path] = "~/.cache/frarch/datasets/oxford_pets/",
    ) -> None:
        if subset not in ["train", "valid"]:
            raise ValueError(f"set must be train or test not {subset}")

        self.root = Path(root).expanduser()
        self.images_root = self.root / "images"

        self.set = subset
        self.transform = transform
        self.target_transform = target_transform

        self.train_lst_path = self.root / "annotations" / "trainval.txt"
        self.valid_lst_path = self.root / "annotations" / "test.txt"

        if download and not self._detect_dataset():
            self._download_dataset()
            self._download_annotations()
        if not self._detect_dataset():
            raise DatasetNotFoundError(
                f"download flag not set and dataset not present in {self.root}"
            )

        self._load_set()

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

    def _download_annotations(self) -> None:
        self._download_file("images")

    def _download_dataset(self) -> None:
        self._download_file("annotations")

    def _download_file(self, url_key: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

        # download train/val images/annotations
        parts = urlparse(urls[url_key])
        filename = Path(parts.path).name
        cached_file = self.root / filename

        if not cached_file.exists():
            logger.info('Downloading: "{}" to {}\n'.format(urls[url_key], cached_file))
            download_url(urls[url_key], cached_file)

        # extract file
        logger.info(f"Extracting tar file {cached_file} to {self.root}")
        tar = tarfile.open(cached_file, "r")
        tar.extractall(self.root)
        tar.close()
        logger.info(f"Done! Removing dached file {cached_file}...")
        cached_file.unlink()

    def _get_file_paths(self) -> List[Path]:
        return list(self.images_root.glob("*.jpg"))

    def _detect_dataset(self) -> bool:
        if not self.root.exists():
            return False
        else:
            num_images = len(self._get_file_paths())
            annotations_present = (
                self.train_lst_path.exists() and self.valid_lst_path.exists()
            )
            return num_images > 0 and annotations_present

    def _load_set(self) -> None:
        path = self.train_lst_path if self.set == "train" else self.valid_lst_path
        with path.open("r") as f:
            self.images = []
            self.classes = set()
            for line in f:
                path, id_, species, breed_id = line.strip().split(" ")
                id_ = int(id_) - 1
                full_path = self.images_root / (path + ".jpg")
                self.images.append((full_path, id_))
                self.classes.add(id_)

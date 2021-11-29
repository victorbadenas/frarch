import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Union

from PIL import Image
from torch.utils.data import Dataset

from frarch.utils.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)


class Caltech101(Dataset):
    def __init__(
        self,
        subset: str = "train",
        transform: Callable = None,
        target_transform: Callable = None,
        root: Union[str, Path] = "./data/",
    ):
        if subset not in ["train", "valid"]:
            raise ValueError(f"set must be train or test not {subset}")

        self.root = Path(root).expanduser()

        self.set = subset
        self.transform = transform
        self.target_transform = target_transform

        self.train_lst_path = self.root / "train.lst"
        self.valid_lst_path = self.root / "valid.lst"
        self.mapper_path = self.root / "classes.json"

        if not self._detect_dataset():
            raise DatasetNotFoundError(
                self.root,
                "Dataset not found at {path}. Please download"
                " it from http://www.vision.caltech.edu/Image_Datasets/Caltech101/ .",
            )

        self._build_and_load_lst()

        logger.info(
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

    def _detect_dataset(self):
        if not self.root.exists():
            return False
        else:
            num_images = len(self._get_file_paths())
            return num_images > 0

    def _build_and_load_lst(self):
        all_paths = self._get_file_paths()
        self._build_and_load_class_map(all_paths)
        self._load_train_test_files(all_paths)

    def _build_and_load_class_map(self, all_paths):
        if not self.mapper_path.exists():
            self._build_class_mapper(all_paths)
        self._load_class_map()

    def _build_class_mapper(self, all_paths):
        classes_set = set(map(lambda path: path.parts[-2], all_paths))
        logger.info(f"found {len(classes_set)} classes.")
        class_mapper = dict(zip(classes_set, range(len(classes_set))))
        logger.info(f"class mapper built: {class_mapper}")
        self._dump_class_map(class_mapper)

    def _load_train_test_files(self, all_paths):
        if not self.train_lst_path.exists() and not self.valid_lst_path.exists():
            self._build_train_test_files(all_paths)
        self._load_set()

    def _build_train_test_files(self, all_paths):
        classes_list = list(map(lambda path: path.parts[-2], all_paths))
        instance_counter = Counter(classes_list)

        train_instances, valid_instances = [], []
        for class_name, count in instance_counter.items():
            class_instances = list(
                filter(lambda x: x.parts[-2] == class_name, all_paths)
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

        self._write_lst(self.train_lst_path, train_instances)
        self._write_lst(self.valid_lst_path, valid_instances)

    @staticmethod
    def _write_lst(path, instances):
        with path.open("w") as f:
            for line in instances:
                f.write(",".join(map(str, line)) + "\n")

    def _get_file_paths(self):
        all_files = list(self.root.glob("*/*.jpg"))
        return list(filter(lambda x: x.parts[-2] != "BACKGROUND_Google", all_files))

    def _load_set(self):
        path = self.train_lst_path if self.set == "train" else self.valid_lst_path
        with path.open("r") as f:
            self.images = []
            for line in f:
                path, label = line.strip().split(",")
                self.images.append((path, int(label)))

    def _dump_class_map(self, class_mapper):
        with self.mapper_path.open("w") as f:
            json.dump(class_mapper, f)

    def _load_class_map(self):
        with self.mapper_path.open("r") as f:
            self.classes = json.load(f)

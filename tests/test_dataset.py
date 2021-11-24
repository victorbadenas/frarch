import shutil
import unittest
from pathlib import Path

import torch

from frarch import datasets
from frarch.utils.exceptions import DatasetNotFoundError

DATA_FOLDER = Path("./tests/data/")


class TestCaltech101(unittest.TestCase):
    MOCK_DATASET_ROOT = DATA_FOLDER / "caltech101"
    trainlst_path = MOCK_DATASET_ROOT / "train.lst"
    validlst_path = MOCK_DATASET_ROOT / "valid.lst"
    classjson_path = MOCK_DATASET_ROOT / "classes.json"

    classes = (DATA_FOLDER / "caltech101_classes.txt").read_text().split(",")

    @classmethod
    def setUpClass(cls):
        if cls.MOCK_DATASET_ROOT.exists():
            shutil.rmtree(cls.MOCK_DATASET_ROOT)
        for c in cls.classes:
            (cls.MOCK_DATASET_ROOT / c).mkdir(parents=True, exist_ok=True)
            for i in range(10):
                (cls.MOCK_DATASET_ROOT / c / f"{i}.jpg").touch()

    @classmethod
    def tearDownClass(cls):
        if cls.MOCK_DATASET_ROOT.exists():
            shutil.rmtree(cls.MOCK_DATASET_ROOT)

    def tearDown(self):
        if self.trainlst_path.exists():
            self.trainlst_path.unlink()
        if self.validlst_path.exists():
            self.validlst_path.unlink()
        if self.classjson_path.exists():
            self.classjson_path.unlink()
        return super().tearDown()

    def test_build_caltech101_train(self):
        dataset = datasets.Caltech101("train", root=self.MOCK_DATASET_ROOT)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset.classes, dict)
        self.assertEquals(len(dataset.classes), 101)
        self.assertEquals(len(dataset.images), 909)
        self.assertEquals(dataset.train_lst_path, self.trainlst_path)
        self.assertEquals(dataset.valid_lst_path, self.validlst_path)
        self.assertEquals(dataset.mapper_path, self.classjson_path)
        self.assertTrue(self.trainlst_path.exists())
        self.assertTrue(self.validlst_path.exists())
        self.assertTrue(self.classjson_path.exists())

    def test_build_caltech101_valid(self):
        dataset = datasets.Caltech101("valid", root=self.MOCK_DATASET_ROOT)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset.classes, dict)
        self.assertEquals(len(dataset.classes), 101)
        self.assertEquals(len(dataset.images), 101)
        self.assertEquals(dataset.train_lst_path, self.trainlst_path)
        self.assertEquals(dataset.valid_lst_path, self.validlst_path)
        self.assertEquals(dataset.mapper_path, self.classjson_path)
        self.assertTrue(self.trainlst_path.exists())
        self.assertTrue(self.validlst_path.exists())
        self.assertTrue(self.classjson_path.exists())

    def test_caltech101_not_valid_subset(self):
        with self.assertRaises(ValueError):
            datasets.Caltech101("nope", root=self.MOCK_DATASET_ROOT)

    def test_caltech101_path_no_files(self):
        with self.assertRaises(DatasetNotFoundError):
            datasets.Caltech101("train", root="./nope/")

    def test_caltech101_get_length(self):
        dataset = datasets.Caltech101("valid", root=self.MOCK_DATASET_ROOT)
        self.assertEqual(len(dataset), 101)

    def test_caltech101_get_num_classes(self):
        dataset = datasets.Caltech101("valid", root=self.MOCK_DATASET_ROOT)
        self.assertEqual(dataset.get_number_classes(), 101)


class TestMit67(unittest.TestCase):
    MOCK_DATASET_ROOT = DATA_FOLDER / "mit67"
    trainlst_path = MOCK_DATASET_ROOT / "train.lst"
    validlst_path = MOCK_DATASET_ROOT / "valid.lst"
    classjson_path = MOCK_DATASET_ROOT / "class_map.json"

    classes = (DATA_FOLDER / "mit67_classes.txt").read_text().split(",")

    @classmethod
    def setUpClass(cls):
        if cls.MOCK_DATASET_ROOT.exists():
            shutil.rmtree(cls.MOCK_DATASET_ROOT)
        for c in cls.classes:
            (cls.MOCK_DATASET_ROOT / "Images" / c).mkdir(parents=True, exist_ok=True)
            for i in range(10):
                (cls.MOCK_DATASET_ROOT / "Images" / c / f"{i}.jpg").touch()

    @classmethod
    def tearDownClass(cls):
        if cls.MOCK_DATASET_ROOT.exists():
            shutil.rmtree(cls.MOCK_DATASET_ROOT)

    def tearDown(self):
        if self.trainlst_path.exists():
            self.trainlst_path.unlink()
        if self.validlst_path.exists():
            self.validlst_path.unlink()
        if self.classjson_path.exists():
            self.classjson_path.unlink()
        return super().tearDown()

    def test_build_mit67_train(self):
        dataset = datasets.Mit67(True, root=self.MOCK_DATASET_ROOT)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset.classes, dict)
        self.assertEquals(len(dataset.classes), 67)
        self.assertEquals(len(dataset.images), 603)
        self.assertEquals(dataset.train_lst_path, self.trainlst_path)
        self.assertEquals(dataset.valid_lst_path, self.validlst_path)
        self.assertEquals(dataset.mapper_path, self.classjson_path)
        self.assertTrue(self.trainlst_path.exists())
        self.assertTrue(self.validlst_path.exists())
        self.assertTrue(self.classjson_path.exists())

    def test_build_mit67_valid(self):
        dataset = datasets.Mit67(False, root=self.MOCK_DATASET_ROOT)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset.classes, dict)
        self.assertEquals(len(dataset.classes), 67)
        self.assertEquals(len(dataset.images), 67)
        self.assertEquals(dataset.train_lst_path, self.trainlst_path)
        self.assertEquals(dataset.valid_lst_path, self.validlst_path)
        self.assertEquals(dataset.mapper_path, self.classjson_path)
        self.assertTrue(self.trainlst_path.exists())
        self.assertTrue(self.validlst_path.exists())
        self.assertTrue(self.classjson_path.exists())

    def test_mit67_path_no_files(self):
        with self.assertRaises(DatasetNotFoundError):
            datasets.Mit67(True, root="./nope/", download=False)

    def test_caltech101_get_length(self):
        dataset = datasets.Mit67(False, root=self.MOCK_DATASET_ROOT)
        self.assertEqual(len(dataset), 67)

    def test_caltech101_get_num_classes(self):
        dataset = datasets.Mit67(False, root=self.MOCK_DATASET_ROOT)
        self.assertEqual(dataset.get_number_classes(), 67)


class TestOxfordPets(unittest.TestCase):
    MOCK_DATASET_ROOT = DATA_FOLDER / "oxfordpets"
    trainlst_path = MOCK_DATASET_ROOT / "annotations" / "trainval.txt"
    validlst_path = MOCK_DATASET_ROOT / "annotations" / "test.txt"

    @classmethod
    def setUpClass(cls):
        if cls.MOCK_DATASET_ROOT.exists():
            shutil.rmtree(cls.MOCK_DATASET_ROOT)
        cls.MOCK_DATASET_ROOT.mkdir(exist_ok=True, parents=True)
        (cls.MOCK_DATASET_ROOT / "images").mkdir(exist_ok=True, parents=True)
        shutil.copytree(
            str(DATA_FOLDER / "oxford_pets_lst"), str(cls.trainlst_path.parent)
        )
        with open(DATA_FOLDER / "oxford_pets_lst" / "trainval.txt") as f:
            for line in f:
                fname = line.split(" ")[0]
                (cls.MOCK_DATASET_ROOT / "images" / f"{fname}.jpg").touch()

    @classmethod
    def tearDownClass(cls):
        if cls.MOCK_DATASET_ROOT.exists():
            shutil.rmtree(cls.MOCK_DATASET_ROOT)

    def test_build_OxfordPets_train(self):
        dataset = datasets.OxfordPets("train", root=self.MOCK_DATASET_ROOT)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset.classes, set)
        self.assertEquals(len(dataset.classes), 37)
        self.assertEquals(len(dataset.images), 3680)
        self.assertEquals(dataset.train_lst_path, self.trainlst_path)
        self.assertEquals(dataset.valid_lst_path, self.validlst_path)
        self.assertTrue(self.trainlst_path.exists())
        self.assertTrue(self.validlst_path.exists())

    def test_build_OxfordPets_valid(self):
        dataset = datasets.OxfordPets("valid", root=self.MOCK_DATASET_ROOT)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset.classes, set)
        self.assertEquals(len(dataset.classes), 37)
        self.assertEquals(len(dataset.images), 3669)
        self.assertEquals(dataset.train_lst_path, self.trainlst_path)
        self.assertEquals(dataset.valid_lst_path, self.validlst_path)
        self.assertTrue(self.trainlst_path.exists())
        self.assertTrue(self.validlst_path.exists())

    def test_OxfordPets_path_no_files(self):
        with self.assertRaises(DatasetNotFoundError):
            datasets.OxfordPets("valid", root="./nope/", download=False)

    def test_OxfordPets_not_valid_subset(self):
        with self.assertRaises(ValueError):
            datasets.OxfordPets("nope", root=self.MOCK_DATASET_ROOT, download=False)

    def test_OxfordPets_get_length(self):
        dataset = datasets.OxfordPets("valid", root=self.MOCK_DATASET_ROOT)
        self.assertEqual(len(dataset), 3669)

    def test_OxfordPets_get_num_classes(self):
        dataset = datasets.OxfordPets("valid", root=self.MOCK_DATASET_ROOT)
        self.assertEqual(dataset.get_number_classes(), 37)


if __name__ == "__main__":
    unittest.main()

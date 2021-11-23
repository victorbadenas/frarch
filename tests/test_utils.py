import shutil
import unittest
from pathlib import Path
from unittest import mock

import torch

from frarch.utils import data, logging

DATA_FOLDER = Path("./tests/data/")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.items = [torch.Tensor(i) for i in range(10)]

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class TestData(unittest.TestCase):
    TMP_FOLDER = Path("./tmp_experiment/")
    TMP_FILE = Path("./tmp")

    def tearDown(self):
        if self.TMP_FILE.exists():
            self.TMP_FILE.unlink()
        if self.TMP_FOLDER.exists():
            shutil.rmtree(self.TMP_FOLDER)
        return super().tearDown()

    def test_create_dataloader(self):
        dataset = DummyDataset()
        dataloader = data.create_dataloader(dataset)
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

    def test_create_dataloader_no_dataset(self):
        with self.assertRaises(ValueError):
            data.create_dataloader(None)

    def test_tensorInDevice(self):
        tensor_data = list(range(10))
        tensor = data.tensorInDevice(tensor_data)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertTupleEqual(tensor.shape, (10,))

    def test_tensorInDevice_cpu(self):
        tensor_data = list(range(10))
        tensor = data.tensorInDevice(tensor_data, device="cpu")
        self.assertEqual(str(tensor.device), "cpu")

    def test_read_file(self):
        filepath = DATA_FOLDER / "text_sample.txt"
        self.assertEqual(data.read_file(filepath), "this is a test file.\n")

    def test_downloadUrl(self):
        data.download_url("https://www.google.com", destination="./tmp")

    def test_downloadUrl_progressbar(self):
        data.download_url(
            "https://www.google.com", destination="./tmp", progress_bar=True
        )

    def test_downloadUrl_no_progressbar(self):
        data.download_url(
            "https://www.google.com", destination="./tmp", progress_bar=False
        )

    def test_build_experiment_structure(self):
        hparam_path = DATA_FOLDER / "hparam_sample.yaml"
        experiment_folder = Path("./tmp_experiment/")
        data.build_experiment_structure(
            hparam_path, experiment_folder=experiment_folder
        )
        self.assertTrue((experiment_folder / "train.yaml").exists())
        self.assertTrue((experiment_folder / "train.log").exists())
        self.assertTrue((experiment_folder / "save").exists())


class TestLogging(unittest.TestCase):
    def test_dummy(self):
        a = 1
        self.assertTrue(1, a)


if __name__ == "__main__":
    unittest.main()

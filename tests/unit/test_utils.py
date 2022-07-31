import logging as logging_module
import shutil
import unittest
from pathlib import Path

import torch

from frarch.utils import data
from frarch.utils import exceptions
from frarch.utils.freezable_module_dict import FreezableModuleDict
from frarch.utils.logging.create_logger import create_logger_file
from tests.mock.mocks import MockClassificationDataset
from tests.mock.mocks import MockModel

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"


def is_file_handler_in_logging(file_path):
    for h in logging_module.root.handlers:
        if h.baseFilename == str(Path(file_path).absolute()):
            return True
    return False


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
        dataset = MockClassificationDataset()
        dataloader = data.create_dataloader(dataset)
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

    def test_create_dataloader_no_dataset(self):
        with self.assertRaises(ValueError):
            data.create_dataloader(None)

    def test_tensorInDevice(self):
        tensor_data = list(range(10))
        tensor = data.tensor_in_device(tensor_data)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertTupleEqual(tensor.shape, (10,))

    def test_tensorInDevice_cpu(self):
        tensor_data = list(range(10))
        tensor = data.tensor_in_device(tensor_data, device="cpu")
        self.assertEqual(str(tensor.device), "cpu")

    def test_read_file(self):
        filepath = DATA_FOLDER / "text_sample.txt"
        self.assertEqual(data.read_file(filepath), "this is a test file.\n")

    def test_downloadUrl(self):
        data.download_url("https://www.google.com", destination="./tmp")

    def test_downloadUrl_progressbar(self):
        file_dest = Path("./tmp")
        data.download_url(
            "https://www.google.com", destination=str(file_dest), progress_bar=True
        )
        self.assertTrue(file_dest.exists())

    def test_downloadUrl_no_progressbar(self):
        file_dest = Path("./tmp")
        data.download_url(
            "https://www.google.com", destination=str(file_dest), progress_bar=False
        )
        self.assertTrue(file_dest.exists())

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
    TMP_LOG = Path("tmp.log")

    def setUp(self):
        if self.TMP_LOG.exists():
            self.TMP_LOG.unlink()
        return super().tearDown()

    def test_create_logger_file(self):
        create_logger_file(self.TMP_LOG)

    def test_create_logger_file_stdout(self):
        create_logger_file(self.TMP_LOG, stdout=True)

    def test_create_logger_file_not_str(self):
        with self.assertRaises(ValueError):
            create_logger_file(0)

    def test_create_logger_debug_not_bool(self):
        with self.assertRaises(ValueError):
            create_logger_file(self.TMP_LOG, debug=0)

    def test_create_logger_stdout_not_bool(self):
        with self.assertRaises(ValueError):
            create_logger_file(self.TMP_LOG, stdout=0)


class TestExceptions(unittest.TestCase):
    def test_dataset_not_found(self):
        path = Path("some_path")
        e = exceptions.DatasetNotFoundError(path)
        self.assertIn(str(path), e.args[0])

    def test_dataset_not_found_raise(self):
        path = Path("some_path")
        with self.assertRaises(exceptions.DatasetNotFoundError):
            raise exceptions.DatasetNotFoundError(path)


class TestFreezableModuleDict(unittest.TestCase):
    modules = {"model": MockModel(), "model2": MockModel()}

    def test_freezable_module_dict_set_modules_no_freeze(self):
        module_dict = FreezableModuleDict(modules=self.modules)
        for module in self.modules:
            self.assertIn(module, module_dict._modules)
            self.assertEqual(getattr(module_dict, module), self.modules[module])

    def test_freezable_module_dict_freeze_layers(self):
        freeze_layers = ["model.fc.weight", "model2.fc.bias"]
        module_dict = FreezableModuleDict(modules=self.modules, freeze=freeze_layers)
        for name, parameter in module_dict.named_parameters():
            if name in freeze_layers:
                self.assertFalse(parameter.requires_grad)
            else:
                self.assertTrue(parameter.requires_grad)


if __name__ == "__main__":
    unittest.main()

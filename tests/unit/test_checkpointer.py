import copy
import json
import shutil
import unittest
from pathlib import Path

import torch

from frarch.modules.checkpointer import Checkpointer

DATA_FOLDER = Path("./tests/data/")


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, input):
        return self.fc(input)


class TestCheckPointer(unittest.TestCase):
    TMP_CKPT_PATH = DATA_FOLDER / "tmp"
    modules = torch.nn.ModuleDict({"model": MockModel(), "model2": MockModel()})

    def tearDown(self):
        if self.TMP_CKPT_PATH.exists():
            shutil.rmtree(self.TMP_CKPT_PATH)
        return super().tearDown()

    def test_init_ok(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH, modules=self.modules, save_best_only=False
        )
        self.assertEqual(self.modules, ckpter.modules)
        self.assertEqual(self.TMP_CKPT_PATH / "save", ckpter.base_path)
        self.assertEqual(ckpter.save_best_only, False)
        self.assertEqual(ckpter.reference_metric, None)
        self.assertEqual(ckpter.mode, "min")
        self.assertTrue(self.TMP_CKPT_PATH.exists())
        self.assertTrue(ckpter.base_path.exists())

    def test_init_path_not_string(self):
        with self.assertRaises(ValueError):
            Checkpointer(save_path=0.0, modules=self.modules)

    def test_init_modules_not_ok(self):
        with self.assertRaises(ValueError):
            Checkpointer(save_path=self.TMP_CKPT_PATH, modules=[MockModel()])

    def test_metadata_in_modules(self):
        nok_modules = copy.deepcopy(self.modules)
        nok_modules["metadata"] = MockModel()
        with self.assertRaises(ValueError):
            Checkpointer(save_path=self.TMP_CKPT_PATH, modules=nok_modules)

    def test_key_not_string_in_modules(self):
        nok_modules = dict(copy.deepcopy(self.modules))
        nok_modules[0] = MockModel()
        with self.assertRaises(ValueError):
            Checkpointer(save_path=self.TMP_CKPT_PATH, modules=nok_modules)

    def test_value_not_module_in_modules(self):
        nok_modules = dict(copy.deepcopy(self.modules))
        nok_modules["module3"] = "not-a-module"
        with self.assertRaises(ValueError):
            Checkpointer(save_path=self.TMP_CKPT_PATH, modules=nok_modules)

    def test_mode_not_valid(self):
        with self.assertRaises(ValueError):
            Checkpointer(
                save_path=self.TMP_CKPT_PATH, modules=self.modules, mode="not-valid"
            )

    def test_save_best_only_no_reference_metric(self):
        with self.assertRaises(ValueError):
            Checkpointer(
                save_path=self.TMP_CKPT_PATH, modules=self.modules, save_best_only=True
            )

    def test_save_initial_weights(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH, modules=self.modules, save_best_only=False
        )
        ckpter.save_initial_weights()
        self.assertTrue((self.TMP_CKPT_PATH / "save" / "initial_weights").exists())
        metadata = read_json(
            self.TMP_CKPT_PATH / "save" / "initial_weights" / "metadata.json"
        )
        self.assertEqual(metadata["epoch"], -1)
        self.assertTrue(
            (self.TMP_CKPT_PATH / "save" / "initial_weights" / "model.pt").exists()
        )
        self.assertTrue(
            (self.TMP_CKPT_PATH / "save" / "initial_weights" / "model2.pt").exists()
        )

    def test_save_end_of_epoch(self):
        Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=self.modules,
            save_best_only=False,
            reference_metric=None,
            mode="min",
        ).save(
            epoch=1, current_step=1000, intra_epoch=False, extra_data={"test": "test"}
        )
        pt_paths = list(self.TMP_CKPT_PATH.glob("**/*.pt"))
        metadata_paths = list(self.TMP_CKPT_PATH.glob("**/metadata.json"))
        self.assertEqual(len(pt_paths), len(self.modules))
        metadata = read_json(metadata_paths[0])
        self.assertFalse(metadata["intra_epoch"])
        self.assertEqual(metadata["step"], 1000)
        self.assertEqual(metadata["epoch"], 1)
        self.assertDictEqual(metadata["extra_info"], {"test": "test"})

    def test_save_intra_epoch(self):
        Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=self.modules,
            save_best_only=False,
            reference_metric=None,
            mode="min",
        ).save(
            epoch=1,
            current_step=1000,
            intra_epoch=True,
        )
        pt_paths = list(self.TMP_CKPT_PATH.glob("**/*.pt"))
        metadata_paths = list(self.TMP_CKPT_PATH.glob("**/metadata.json"))
        self.assertEqual(len(pt_paths), len(self.modules))
        metadata = read_json(metadata_paths[0])
        self.assertTrue(metadata["intra_epoch"])
        self.assertEqual(metadata["step"], 1000)
        self.assertEqual(metadata["epoch"], 1)

    def test_save_extradata(self):
        Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=self.modules,
            save_best_only=False,
            reference_metric=None,
            mode="min",
        ).save(
            epoch=1, current_step=1000, intra_epoch=True, extra_data={"test": "test"}
        )
        metadata_paths = list(self.TMP_CKPT_PATH.glob("**/metadata.json"))
        metadata = read_json(metadata_paths[0])
        self.assertDictEqual(metadata["extra_info"], {"test": "test"})

    def test_save_metric(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=self.modules,
            save_best_only=True,
            reference_metric="metric",
            mode="min",
        )
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.5)
        self.assertEqual(ckpter.best_metric, 0.5)

    def test_save_metric_update_min(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=self.modules,
            save_best_only=True,
            reference_metric="metric",
            mode="min",
        )
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.5)
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.1)
        self.assertEqual(ckpter.best_metric, 0.1)

    def test_save_metric_update_max(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=self.modules,
            save_best_only=True,
            reference_metric="metric",
            mode="max",
        )
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.5)
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.1)
        self.assertEqual(ckpter.best_metric, 0.5)

    def test_load_checkpoint(self):
        modules = copy.deepcopy(self.modules)
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=modules,
            save_best_only=True,
            reference_metric="metric",
            mode="max",
        )
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.5)
        modules.model.fc = torch.nn.Linear(2, 1)
        ckpter.load(mode="last")
        self.assertTrue((self.modules.model.fc.weight == modules.model.fc.weight).all())

    def test_load_checkpoint_map_location(self):
        modules = copy.deepcopy(self.modules)
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH,
            modules=modules,
            save_best_only=True,
            reference_metric="metric",
            mode="max",
        )
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.5)
        modules.model.fc = torch.nn.Linear(2, 1)
        ckpter.load(mode="last", map_location="cpu")
        self.assertTrue((self.modules.model.fc.weight == modules.model.fc.weight).all())

    def test_properies_end_of_epoch(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH, modules=self.modules, save_best_only=False
        )
        ckpter.save(epoch=1, current_step=1000, intra_epoch=False, metric=0.5)
        self.assertEqual(ckpter._is_intraepoch(), False)
        self.assertEqual(ckpter.current_epoch, 1)
        self.assertEqual(ckpter.next_epoch, 2)
        self.assertEqual(ckpter.step, 0)

    def test_properies_intra_epoch(self):
        ckpter = Checkpointer(
            save_path=self.TMP_CKPT_PATH, modules=self.modules, save_best_only=False
        )
        ckpter.save(
            epoch=1,
            current_step=1000,
            intra_epoch=True,
        )
        self.assertEqual(ckpter._is_intraepoch(), True)
        self.assertEqual(ckpter.current_epoch, 1)
        self.assertEqual(ckpter.next_epoch, 1)
        self.assertEqual(ckpter.step, 1000)


def read_json(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    unittest.main()

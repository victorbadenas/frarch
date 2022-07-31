import unittest
from pathlib import Path

import torch

from frarch.train.base_trainer import BaseTrainer
from frarch.train.classifier_trainer import ClassifierTrainer
from tests.mock.mocks import MockClassificationDataset
from tests.mock.mocks import MockModel

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"


class TestClassifierTrainer(ClassifierTrainer):
    def _forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.modules.model(inputs)

    def _compute_loss(self, predictions, batch, stage):
        _, labels = batch
        labels = labels.to(self.device)
        return self.hparams["loss"](predictions, labels)


class TestTrainers(unittest.TestCase):
    model = MockModel()
    train_dataset = MockClassificationDataset()
    test_dataset = MockClassificationDataset()
    opt_class = torch.optim.Adam

    def test_init(self):
        BaseTrainer({"model": self.model}, self.opt_class, {"noprogressbar": True})

    def test_init_ckpt_interval_negative(self):
        with self.assertRaises(ValueError):
            BaseTrainer(
                {"model": self.model}, self.opt_class, {"ckpt_interval_minutes": -1}
            )

    def test_ClassifierTrainer_init(self):
        TestClassifierTrainer(
            {"model": self.model}, self.opt_class, {"noprogressbar": True}
        )

    def test_ClassifierTrainer_fit(self):
        trainer = TestClassifierTrainer(
            {"model": self.model},
            self.opt_class,
            {"epochs": 1, "loss": torch.nn.CrossEntropyLoss(), "noprogressbar": True},
        )
        trainer.fit(
            train_set=self.train_dataset,
            valid_set=self.test_dataset,
        )

    def test_ClassifierTrainer_fit_epochs_not_specified(self):
        trainer = TestClassifierTrainer(
            {"model": self.model},
            self.opt_class,
            {"loss": torch.nn.CrossEntropyLoss(), "noprogressbar": True},
        )
        with self.assertRaises(KeyError):
            trainer.fit(
                train_set=self.train_dataset,
                valid_set=self.test_dataset,
            )

    def test_ClassifierTrainer_fit_loss_not_specified(self):
        trainer = TestClassifierTrainer(
            {"model": self.model}, self.opt_class, {"epochs": 1, "noprogressbar": True}
        )
        with self.assertRaises(KeyError):
            trainer.fit(
                train_set=self.train_dataset,
                valid_set=self.test_dataset,
            )


if __name__ == "__main__":
    unittest.main()

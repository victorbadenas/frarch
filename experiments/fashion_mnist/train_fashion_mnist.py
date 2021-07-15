"""
Script to train a model to classify Fashion-MNist dataset.

:Description: script to train the fashionMNist task

:Authors: victor badenas (victor.badenas@gmail.com)

:Version: 0.1.0
:Created on: 01/06/2021 11:00
"""

__title__ = "train_fashion_mnist"
__version__ = "0.1.0"
__author__ = "victor badenas"

import logging
import os
import sys
from pathlib import Path
from pprint import pprint

import torch
from hyperpyyaml import load_hyperpyyaml

import frarch as fr

logger = logging.getLogger(__name__)

from frarch.utils.data import build_experiment_structure
from frarch.utils.stages import Stage


class FMNISTTrainer(fr.train.ClassifierTrainer):
    def forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        embeddings = self.modules.model(inputs)
        return self.modules.classifier(embeddings)

    def compute_loss(self, predictions, batch, stage):
        _, labels = batch
        loss = self.hparams["loss"](predictions, labels)
        if stage == Stage.VALID and "error_metrics" in self.hparams:
            self.hparams["error_metrics"].update(predictions, labels)
        return loss

    def on_stage_start(self, stage, loss=None, epoch=None):
        if stage == Stage.VALID:
            self.hparams["error_metrics"].reset()
            if self.debug:
                metrics = self.hparams["error_metrics"].get_metrics(mode="mean")

    def on_stage_end(self, stage, loss=None, epoch=None):
        if stage == Stage.VALID:
            metrics = self.hparams["error_metrics"].get_metrics(mode="mean")
            metrics_string = "".join([f"{k}=={v:.4f}" for k, v in metrics.items()])
            logging.info(f"epoch {epoch}: train_loss {self.avg_train_loss:.4f} validation_loss {loss:.4f} metrics: {metrics_string}")
            if self.checkpointer is not None:
                metrics["train_loss"] = self.avg_train_loss
                metrics["val_loss"] = loss
                self.checkpointer.save(**metrics, epoch=self.current_epoch)


if __name__ == "__main__":
    hparam_file, args = fr.parse_arguments()

    with open(hparam_file, "r") as hparam_file_handler:
        hparams = load_hyperpyyaml(
            hparam_file_handler, args, overrides_must_match=False
        )

    build_experiment_structure(
        hparam_file,
        overrides=args,
        experiment_folder=hparams["experiment_folder"],
        debug=hparams["debug"],
    )

    trainer = FMNISTTrainer(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    trainer.fit(
        train_set=hparams["train_dataset"],
        valid_set=hparams["valid_dataset"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

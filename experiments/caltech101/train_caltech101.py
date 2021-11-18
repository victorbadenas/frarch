"""
Script to train a model to classify caltech101 dataset.

:Description: script to train the caltech101 task

:Authors: victor badenas (victor.badenas@gmail.com)

:Version: 0.1.0
:Created on: mie 27 oct 2021 16:19:42 CEST
"""

__title__ = "train_caltech101"
__version__ = "0.1.0"
__author__ = "victor badenas"

import logging

import torch
from hyperpyyaml import load_hyperpyyaml

import frarch as fr

logger = logging.getLogger(__name__)

from frarch.utils.data import build_experiment_structure
from frarch.utils.stages import Stage


class Caltech101Trainer(fr.train.ClassifierTrainer):
    def __init__(self, *args, **kwargs):
        super(Caltech101Trainer, self).__init__(*args, **kwargs)
        if "padding" in self.hparams:
            if self.hparams["padding"] == "valid":
                self.change_model_padding()
            elif self.hparams["padding"] == "same":
                logger.info("padding not changed. Defaulting to same.")
            else:
                logger.warning(
                    "padding configuration not understood. Defaulting to same."
                )

    def change_model_padding(self):
        for layer_name, layer in self.modules.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                padding_conf = self.hparams["padding"]
                logger.info(
                    f"Changing {layer_name}'s padding from same to {padding_conf}"
                )
                layer._reversed_padding_repeated_twice = (0, 0, 0, 0)
                layer.padding = (0, 0)
        self.modules.model.avgpool.output_size = (3, 3)
        self.modules.model.classifier[0] = torch.nn.Linear(512 * 3 * 3, 4096)

    def forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.modules.model(inputs)

    def compute_loss(self, predictions, batch, stage):
        _, labels = batch
        labels = labels.to(self.device)
        loss = self.hparams["loss"](predictions, labels)
        self.hparams["metrics"].update(predictions, labels)
        return loss

    def on_stage_start(self, stage, loss=None, epoch=None):
        self.hparams["metrics"].reset()
        if self.debug:
            metrics = self.hparams["metrics"].get_metrics(mode="mean")
            logger.debug(metrics)

    def on_stage_end(self, stage, loss=None, epoch=None):
        metrics = self.hparams["metrics"].get_metrics(mode="mean")
        metrics_string = "".join([f"{k}={v:.4f}" for k, v in metrics.items()])

        if stage == Stage.TRAIN:
            self.train_metrics = metrics
            self.train_metrics_string = metrics_string

        elif stage == Stage.VALID:
            logger.info(
                f"epoch {epoch}: train_loss={self.avg_train_loss:.4f}"
                f" validation_loss={loss:.4f}"
                f" train_metrics: {self.train_metrics_string}"
                f" validation_metrics: {metrics_string}"
            )
            if self.checkpointer is not None:
                metrics["train_loss"] = self.avg_train_loss
                metrics["val_loss"] = loss
                self.checkpointer.save(
                    epoch=self.current_epoch,
                    current_step=self.step,
                    extra_data={"train": self.train_metrics, "val": metrics},
                    **metrics,
                )
            with open(self.checkpointer.base_path / "metrics.csv", "a+") as f:
                if self.current_epoch == 0:
                    f.write("epoch,train_loss,val_loss,train_err,val_err\n")
                f.write(
                    f"{epoch},{self.avg_train_loss},{loss},"
                    f"{self.train_metrics[self.checkpointer.reference_metric]},"
                    f"{metrics[self.checkpointer.reference_metric]}\n"
                )


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

    trainer = Caltech101Trainer(
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

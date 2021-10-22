"""
Script to train a model to classify mit67 dataset.

:Description: script to train the mit67 task

:Authors: victor badenas (victor.badenas@gmail.com)

:Version: 0.1.0
:Created on: 01/06/2021 11:00
"""

__title__ = "mit67"
__version__ = "0.1.0"
__author__ = "victor badenas"

import logging

from hyperpyyaml import load_hyperpyyaml

import frarch as fr

logger = logging.getLogger(__name__)

from frarch.utils.data import build_experiment_structure
from frarch.utils.stages import Stage


class Mit67Trainer(fr.train.ClassifierTrainer):
    def forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        embeddings = self.modules.model(inputs)
        return self.modules.classifier(embeddings)

    def compute_loss(self, predictions, batch, stage):
        _, labels = batch
        labels = labels.to(self.device)
        loss = self.hparams["loss"](predictions, labels)
        # if stage == Stage.VALID and "metrics" in self.hparams:
        self.hparams["metrics"].update(predictions, labels)
        return loss

    def on_stage_start(self, stage, loss=None, epoch=None):
        # if stage == Stage.VALID:
        self.hparams["metrics"].reset()
        if self.debug:
            metrics = self.hparams["metrics"].get_metrics(mode="mean")
            logger.debug(metrics)

    def on_stage_end(self, stage, loss=None, epoch=None):
        # if stage == Stage.VALID:
        metrics = self.hparams["metrics"].get_metrics(mode="mean")
        metrics_string = "".join([f"{k}=={v:.4f}" for k, v in metrics.items()])
        logging.info(
            f"epoch {epoch}: train_loss {self.avg_train_loss:.4f}"
            f" validation_loss {loss:.4f} metrics: {metrics_string}"
        )
        if stage == Stage.VALID:
            if self.checkpointer is not None:
                metrics["train_loss"] = self.avg_train_loss
                metrics["val_loss"] = loss
                self.checkpointer.save(
                    **metrics, epoch=self.current_epoch, current_step=self.step
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

    trainer = Mit67Trainer(
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

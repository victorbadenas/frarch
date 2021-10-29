"""
Base class for classifier network training.

:Description: base trainer class for training a classifier network

:Authors: victor badenas (victor.badenas@gmail.com)

:Version: 0.1.0
"""

__title__ = "classifier_trainer"
__version__ = "0.1.0"
__author__ = "victor badenas"

import logging
import time

import torch

from frarch.utils.data import create_dataloader
from frarch.utils.stages import Stage

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class ClassifierTrainer(BaseTrainer):
    def fit(
        self,
        train_set,
        valid_set=None,
        train_loader_kwargs: dict = None,
        valid_loader_kwargs: dict = None,
    ):
        if train_loader_kwargs is None:
            train_loader_kwargs = {}
        if valid_loader_kwargs is None:
            valid_loader_kwargs = {}

        if not isinstance(train_set, torch.utils.data.DataLoader):
            train_set = create_dataloader(train_set, **train_loader_kwargs)
        if valid_set is not None and not isinstance(
            valid_set, torch.utils.data.DataLoader
        ):
            valid_set = create_dataloader(valid_set, **valid_loader_kwargs)

        self.on_fit_start()

        for self.current_epoch in range(self.start_epoch, self.hparams["epochs"]):

            self.on_stage_start(Stage.TRAIN, self.current_epoch)
            self.modules.train()

            last_ckpt_time = time.time()

            t = self.get_iterable(
                train_set,
                desc=f"Epoch {self.current_epoch} train",
                initial=self.step,
                dynamic_ncols=True,
            )
            for batch in t:
                self.step += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                self.update_progress(
                    t, self.step, stage="train", train_loss=self.avg_train_loss
                )

                if not (self.step % self.train_interval):
                    self.on_train_interval(self.current_epoch)

                if self.debug and self.step >= self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes is not None
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    self.save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

            if not self.noprogressbar:
                t.close()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, self.current_epoch)

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, self.current_epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                valid_step = 0
                with torch.no_grad():
                    t = self.get_iterable(
                        valid_set,
                        desc=f"Epoch {self.current_epoch} valid",
                        dynamic_ncols=True,
                    )
                    for batch in t:
                        valid_step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(loss, avg_valid_loss)
                        self.update_progress(
                            t, valid_step, stage="valid", valid_loss=avg_valid_loss
                        )

                    if not self.noprogressbar:
                        t.close()

                    # Only run validation "on_stage_end" on main process
                    self.on_stage_end(Stage.VALID, avg_valid_loss, self.current_epoch)
            self.step = 0
            self.avg_train_loss = 0.0
        self.on_fit_end()

    def get_iterable(self, dataset: torch.utils.data.DataLoader, **kwargs):
        if not self.noprogressbar:
            from tqdm import tqdm

            t = tqdm(dataset, **kwargs)
            return t
        else:
            logger.warning(f"Running {self.__class__.__name__} without tqdm")
            return dataset

    def update_progress(self, iterable, step, stage=None, **kwargs):
        if self.noprogressbar:
            if not step % self.train_interval or (step >= len(iterable)) or (step == 1):
                self.update_progress_console(iterable, step, stage=stage, **kwargs)
        else:
            self.update_progress_tqdm(iterable, **kwargs)

    def update_progress_tqdm(self, iterable, **kwargs):
        if "metrics" in self.hparams:
            iterable.set_postfix(
                **kwargs,
                **self.hparams["metrics"].get_metrics(mode="mean"),
            )
        else:
            iterable.set_postfix(**kwargs)

    def update_progress_console(self, iterable, step, stage=None, **kwargs):
        kwargs_string = ", ".join([f"{k}={v:.4f}" for k, v in kwargs.items()])
        if "metrics" in self.hparams:
            metrics = self.hparams["metrics"].get_metrics(mode="mean")
            metrics_string = ",".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        else:
            metrics_string = ""

        print(
            f"Epoch {self.current_epoch} {stage}: step {step}/{len(iterable)}"
            f" -> {kwargs_string}, {metrics_string}"
        )

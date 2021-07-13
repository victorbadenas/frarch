"""
.. module:: classifier_trainer
classifier_trainer
*************
:Description: base trainer class for training a classifier network

:Authors: victor badenas (victor.badenas@gmail.com)

:Version: 0.1.0
"""

__title__ = "classifier_trainer"
__version__ = "0.1.0"
__author__ = "victor badenas"

import logging
import sys
import time

import torch
from tqdm import tqdm

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

        for epoch in range(self.current_epoch, self.hparams["epochs"]):

            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()

            last_ckpt_time = time.time()

            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=self.hparams["noprogressbar"],
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                    t.set_postfix(train_loss=self.avg_train_loss)

                    if not (self.step % self.train_interval):
                        self.on_train_interval(epoch)

                    if self.debug and self.step >= self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set,
                        dynamic_ncols=True,
                        disable=self.hparams["noprogressbar"],
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)

        self.on_fit_end()

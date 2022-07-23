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
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from frarch.utils.data import create_dataloader
from frarch.utils.enums.stages import Stage

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class ClassifierTrainer(BaseTrainer):
    """Trainer class for classifiers.

    Args:
        modules (Mapping[str, torch.nn.Module]): trainable modules in the training.
        opt_class (Type[torch.optim.Optimizer]): optimizer class for training.
        hparams (Mapping[str, Any]): hparams dict-like structure from hparams file.
        checkpointer (Optional[Checkpointer], optional): Checkpointer class for saving
            the model and the hyperparameters needed. If None, no checkpoints are saved.
            Defaults to None.

    Raises:
        ValueError: ckpt_interval_minutes must be > 0 or None
        SystemError: Python version not supported. Python version must be >= 3.7
    """

    def fit(
        self,
        train_set: Union[Dataset, DataLoader],
        valid_set: Optional[Union[Dataset, DataLoader]] = None,
        train_loader_kwargs: dict = None,
        valid_loader_kwargs: dict = None,
    ) -> None:
        """Fit the modules to the dataset. Main function of the Trainer class.

        Args:
            train_set (Union[Dataset, DataLoader]): dataset for training.
            valid_set (Optional[Union[Dataset, DataLoader]], optional): dataset for
                validation. If not provided, validation will not be performed.
                Defaults to None.
            train_loader_kwargs (dict, optional): optional kwargs for train dataloader.
                Defaults to None.
            valid_loader_kwargs (dict, optional): optional kwargs for valid dataloader.
                Defaults to None.
        """
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

        self._on_fit_start()

        for self.current_epoch in range(self.start_epoch, self.hparams["epochs"]):

            self._on_stage_start(Stage.TRAIN, self.current_epoch)
            self.modules.train()

            last_ckpt_time = time.time()

            t = self._get_iterable(
                train_set,
                desc=f"Epoch {self.current_epoch} train",
                initial=self.step,
                dynamic_ncols=True,
            )
            for batch in t:
                self.step += 1
                loss = self._fit_batch(batch)
                self.avg_train_loss = self._update_average(loss, self.avg_train_loss)
                self._update_progress(
                    t, self.step, stage="train", train_loss=self.avg_train_loss
                )

                if not (self.step % self.train_interval):
                    self._on_train_interval(self.current_epoch)

                if self.debug and self.step >= self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes is not None
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

            if not self.noprogressbar:
                t.close()

            # Run train "on_stage_end" on all processes
            self._on_stage_end(Stage.TRAIN, self.avg_train_loss, self.current_epoch)

            # Validation stage
            if valid_set is not None:
                self._on_stage_start(Stage.VALID, self.current_epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                valid_step = 0
                with torch.no_grad():
                    t = self._get_iterable(
                        valid_set,
                        desc=f"Epoch {self.current_epoch} valid",
                        dynamic_ncols=True,
                    )
                    for batch in t:
                        valid_step += 1
                        loss = self._evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self._update_average(loss, avg_valid_loss)
                        self._update_progress(
                            t, valid_step, stage="valid", valid_loss=avg_valid_loss
                        )
                        if self.debug and self.step >= self.debug_batches:
                            break

                    if not self.noprogressbar:
                        t.close()

                    # Only run validation "on_stage_end" on main process
                    self._on_stage_end(Stage.VALID, avg_valid_loss, self.current_epoch)
            self.step = 0
            self.avg_train_loss = 0.0
        self._on_fit_end(self.current_epoch)

    def _get_iterable(self, dataset: torch.utils.data.DataLoader, **kwargs) -> Iterable:
        if not self.noprogressbar:
            from tqdm import tqdm

            t = tqdm(dataset, **kwargs)
            return t
        else:
            logger.warning(f"Running {self.__class__.__name__} without tqdm")
            return dataset

    def _update_progress(
        self, iterable: Iterable, step: int, stage: Stage, **kwargs
    ) -> None:
        if self.noprogressbar:
            if not step % self.train_interval or (step >= len(iterable)) or (step == 1):
                self._update_progress_console(iterable, step, stage=stage, **kwargs)
        else:
            self._update_progress_tqdm(iterable, **kwargs)

    def _update_progress_tqdm(self, iterable: Iterable, **kwargs) -> None:
        if "metrics" in self.hparams:
            iterable.set_postfix(
                **kwargs,
                **self.hparams["metrics"].get_metrics(mode="mean"),
            )
        else:
            iterable.set_postfix(**kwargs)

    def _update_progress_console(
        self, iterable: Iterable, step: int, stage: Stage, **kwargs
    ) -> None:
        kwargs_string = ", ".join([f"{k}={v:.4f}" for k, v in kwargs.items()])
        if "metrics" in self.hparams:
            metrics = self.hparams["metrics"].get_metrics(mode="mean")
            metrics_string = ",".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        else:
            metrics_string = ""

        logger.info(
            f"Epoch {self.current_epoch} {stage}: step {step}/{len(iterable)}"
            f" -> {kwargs_string}, {metrics_string}"
        )

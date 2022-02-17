"""
class definition of a base trainer.

:Description: base trainer class

:Authors: victor badenas (victor.badenas@gmail.com)

:Version: 0.1.0
"""

__title__ = "base_trainer"
__version__ = "0.1.0"
__author__ = "victor badenas"

import logging
import sys
from typing import Any, Mapping, Optional, Type, Union

import torch
from torch.utils.data import DataLoader, Dataset

from frarch.modules import Checkpointer
from frarch.utils.stages import Stage

logger = logging.getLogger(__name__)
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 7

default_values = {
    "debug": False,
    "debug_batches": 2,
    "device": "cpu",
    "nonfinite_patience": 3,
    "noprogressbar": False,
    "ckpt_interval_minutes": None,
    "train_interval": 10,
}


class BaseTrainer:
    """Abstract class for trainer managers.

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

    def __init__(
        self,
        modules: Mapping[str, torch.nn.Module],
        opt_class: Type[torch.optim.Optimizer],
        hparams: Mapping[str, Any],
        checkpointer: Optional[Checkpointer] = None,
    ) -> None:
        self.hparams = hparams
        self.opt_class = opt_class
        self.checkpointer = checkpointer

        for name, value in default_values.items():
            if name in self.hparams:
                if value != hparams[name]:
                    logger.info(
                        f"Parameter {name} overriden from"
                        f" default value and set to {hparams[name]}"
                    )
                setattr(self, name, hparams[name])
            else:
                setattr(self, name, value)

        if self.ckpt_interval_minutes is not None:
            if self.ckpt_interval_minutes <= 0:
                raise ValueError("ckpt_interval_minutes must be > 0 or None")

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            msg = (
                f"Detected Python {sys.version_info.major}.{sys.version_info.minor}."
                f"Python >= {PYTHON_VERSION_MAJOR}.{PYTHON_VERSION_MINOR} is required"
            )
            logger.error(msg)
            raise SystemError(msg)

        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0

    def __call__(self, *args, **kwargs):
        """Alias for fit."""
        return self.fit(*args, **kwargs)

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
        raise NotImplementedError

    def _on_fit_start(self):
        # Initialize optimizers
        self._init_optimizers()

        # set first epoch index
        self.start_epoch = 0

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            if self.checkpointer.exists_checkpoint():
                logger.info("loading last checkpoint")
                if self.checkpointer.load(mode="last"):
                    self.start_epoch = self.checkpointer.next_epoch
                    self.step = self.checkpointer.step
                    logger.info(
                        "resuming training from epoch "
                        f"{self.start_epoch} in step: {self.step}"
                    )

        if self.start_epoch == 0:
            self._save_initial_weights()

    def _save_initial_weights(self):
        if self.checkpointer is not None:
            self.checkpointer.save_initial_weights()

    def _init_optimizers(self):
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.parameters())

    def _on_fit_end(self, epoch: Optional[int] = None):
        pass

    def _on_stage_start(self, stage: Stage, epoch: Optional[int] = None):
        pass

    def _on_stage_end(self, stage: Stage, loss=None, epoch: Optional[int] = None):
        pass

    def _on_train_interval(self, epoch: Optional[int] = None):
        pass

    def _save_intra_epoch_ckpt(self):
        raise NotImplementedError

    def _forward(self, batch: torch.Tensor, stage: Stage):
        raise NotImplementedError

    def _evaluate_batch(self, batch: torch.Tensor, stage: Stage):
        out = self._forward(batch, stage=stage)
        loss = self._compute_loss(out, batch, stage=stage)
        return loss.detach().cpu()

    def _compute_loss(
        self, predictions: torch.Tensor, batch: torch.Tensor, stage: Stage
    ):
        raise NotImplementedError

    def _fit_batch(self, batch: torch.Tensor):
        self.optimizer.zero_grad()
        outputs = self._forward(batch, Stage.TRAIN)
        loss = self._compute_loss(outputs, batch, Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()

    def _update_average(self, loss: torch.Tensor, avg_loss: torch.Tensor):
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

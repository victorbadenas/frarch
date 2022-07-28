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
from typing import Any, Mapping, Optional, Type, Union, List

import torch
from torch.nn import ModuleDict, Module
from torch.utils.data import DataLoader, Dataset

from frarch.modules.checkpointer import Checkpointer
from frarch.utils.enums.stages import Stage
from frarch.utils.freezable_module_dict import FreezableModuleDict

logger = logging.getLogger(__name__)


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

    modules: ModuleDict
    debug: bool = False
    debug_batches: int = 2
    device: str = "cpu"
    nonfinite_patience: int = 3
    noprogressbar: bool = False
    ckpt_interval_minutes: Optional[int] = None
    train_interval: int = 10

    def __init__(
        self,
        modules: Mapping[str, Module],
        opt_class: Type[torch.optim.Optimizer],
        hparams: Mapping[str, Any],
        checkpointer: Optional[Checkpointer] = None,
        freeze_layers: Optional[List[str]] = None,
    ) -> None:
        for name in hparams:
            if hasattr(self, name):
                if getattr(self, name) != hparams[name]:
                    logger.info(
                        f"Parameter {name} overriden from"
                        f" default value and set to {hparams[name]}"
                    )
                setattr(self, name, hparams[name])

        self.hparams = hparams
        self.opt_class: Type[torch.optim.Optimizer] = opt_class
        self.checkpointer = checkpointer
        self.modules = FreezableModuleDict(modules=modules, freeze=freeze_layers).to(
            self.device
        )

        if self.ckpt_interval_minutes is not None:
            if self.ckpt_interval_minutes <= 0:
                raise ValueError("ckpt_interval_minutes must be > 0 or None")

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0

    def __call__(self, *args, **kwargs) -> None:
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

    def _on_fit_start(self) -> None:
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

    def _save_initial_weights(self) -> None:
        if self.checkpointer is not None:
            self.checkpointer.save_initial_weights()

    def _init_optimizers(self) -> None:
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.parameters())

    def _on_fit_end(self, epoch: int) -> None:
        """Perform operation before fit exits.

        Args:
            epoch (int): int epoch index.
        """

    def _on_stage_start(self, stage: Stage, epoch: int) -> None:
        """Perform operation before a stage starts.

        Args:
            stage (Stage): stage where function has been called.
            epoch (int): int epoch index.
        """

    def _on_stage_end(self, stage: Stage, loss: torch.Tensor, epoch: int) -> None:
        """Perform operation before a stage ends.

        Args:
            stage (Stage): stage where function has been called.
            loss (torch.Tensor): loss tensor from `_fit_batch`
            epoch (int): int epoch index.
        """

    def _on_train_interval(self, epoch: int) -> None:
        """Perform operation every `self.train_interval` batches.

        Args:
            epoch (int): int epoch index.
        """

    def _save_intra_epoch_ckpt(self) -> None:
        raise NotImplementedError

    def _forward(self, batch: torch.Tensor, stage: Stage) -> torch.Tensor:
        raise NotImplementedError

    def _evaluate_batch(self, batch: torch.Tensor, stage: Stage) -> torch.Tensor:
        out = self._forward(batch, stage=stage)
        loss = self._compute_loss(out, batch, stage=stage)
        return loss.detach().cpu()

    def _compute_loss(
        self, predictions: torch.Tensor, batch: torch.Tensor, stage: Stage
    ) -> torch.Tensor:
        raise NotImplementedError

    def _fit_batch(self, batch: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        outputs = self._forward(batch, Stage.TRAIN)
        loss = self._compute_loss(outputs, batch, Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()

    def _update_average(
        self, loss: torch.Tensor, avg_loss: torch.Tensor
    ) -> torch.Tensor:
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

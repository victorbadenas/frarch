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

import torch

from frarch.utils.stages import Stage

logger = logging.getLogger(__name__)
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 6

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
    def __init__(self, modules, opt_class, hparams, checkpointer=None):
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
        return self.fit(*args, **kwargs)

    def on_fit_start(self):
        # Initialize optimizers
        self.init_optimizers()

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
            self.save_initial_weights()

    def save_initial_weights(self):
        if self.checkpointer is not None:
            self.checkpointer.save_initial_weights()

    def init_optimizers(self):
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.parameters())

    def on_fit_end(self, epoch=None):
        pass

    def on_stage_start(self, stage, epoch=None):
        pass

    def on_stage_end(self, stage, loss=None, epoch=None):
        pass

    def on_train_interval(self, epoch=None):
        pass

    def save_intra_epoch_ckpt(self):
        raise NotImplementedError

    def forward(self, batch, stage):
        raise NotImplementedError

    def evaluate_batch(self, batch, stage):
        out = self.forward(batch, stage=stage)
        loss = self.compute_loss(out, batch, stage=stage)
        return loss.detach().cpu()

    def compute_loss(self, predictions, batch, stage):
        raise NotImplementedError

    def fit_batch(self, batch):
        self.optimizer.zero_grad()
        outputs = self.forward(batch, Stage.TRAIN)
        loss = self.compute_loss(outputs, batch, Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()

    def update_average(self, loss, avg_loss):
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

    def fit(
        self,
        train_set,
        valid_set=None,
        train_loader_kwargs: dict = None,
        valid_loader_kwargs: dict = None,
    ):
        raise NotImplementedError

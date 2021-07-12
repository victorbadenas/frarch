"""
.. module:: classifier_trainer
classifier_trainer
*************
:Description: base trainer class for training a classifier network
    
:Authors: victor badenas (victor.badenas@gmail.com)
    
:Version: 0.1.0
"""

__title__ = 'classifier_trainer'
__version__ = '0.1.0'
__author__ = 'victor badenas'

import sys
import time
import torch
import logging
from tqdm import tqdm
from frarch.utils.data import create_dataloader
from frarch.utils.stages import Stage
from frarch.utils.logging import create_logger

logger = logging.getLogger(__name__)
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 6

default_values = {
    "debug": False,
    "debug_batches": 2,
    "device": "cpu",
    "nonfinite_patience": 3,
    "noprogressbar": False,
    "ckpt_interval_minutes": 0,
    "train_interval": 10
}

class ClassifierTrainer:
    def __init__(self, modules, opt_class, hparams, checkpointer=None):
        if hparams['log_file'] is None:
            hparams['log_file'] = f'results/{hparams.get("experiment_name", "debug")}/train.log'
        create_logger(hparams['log_file'], debug=hparams['debug'], stdout=hparams['debug'])
        self.hparams = hparams
        self.opt_class = opt_class
        self.checkpointer = checkpointer

        for name, value in default_values.items():
            if name in self.hparams:
                if value != hparams[name]:
                    logger.info(f'Parameter {name} overriden from default value and set to {hparams[name]}')
                    setattr(self, name, hparams[name])
            else:
                setattr(self, name, value)

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            msg = f"Detected Python {sys.version_info.major}.{sys.version_info.minor}. Python >= {PYTHON_VERSION_MAJOR}.{PYTHON_VERSION_MINOR} is required"
            logger.error(msg)
            raise SystemError(msg)

        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def on_fit_start(self):
        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # set first epoch index
        self.current_epoch = 0

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            # TODO: load latest checkpoint
            pass

    def init_optimizers(self):
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.parameters())

    def on_fit_end(self, stage, epoch=None):
        pass

    def on_stage_start(self, stage, epoch=None):
        pass

    def on_stage_end(self, stage, loss=None, epoch=None):
        pass

    def on_train_interval(self, epoch=None):
        pass

    def forward(self, batch, stage):
        raise NotImplementedError

    def evaluate_batch(self, batch, stage):
        out = self.forward(batch, stage=stage)
        loss = self.compute_loss(out, batch, stage=stage)
        return loss.detach().cpu()

    def compute_loss(self, predictions, batch, stage):
        raise NotImplementedError

    def fit_batch(self, batch):
        outputs = self.forward(batch, Stage.TRAIN)
        loss = self.compute_loss(outputs, batch, Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def update_average(self, loss, avg_loss):
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

    def fit(self, train_set,
            valid_set=None,
            train_loader_kwargs:dict=None,
            valid_loader_kwargs:dict=None
        ):

        if train_loader_kwargs is None:
            train_loader_kwargs = {}
        if valid_loader_kwargs is None:
            valid_loader_kwargs = {}

        if not isinstance(train_set, torch.utils.data.DataLoader):
            train_set = create_dataloader(train_set, **train_loader_kwargs)
        if valid_set is not None and not isinstance(valid_set, torch.utils.data.DataLoader):
            valid_set = create_dataloader(valid_set, **valid_loader_kwargs)

        self.on_fit_start()

        for epoch in range(self.current_epoch, self.hparams['epochs']):

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
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
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
                        valid_set, dynamic_ncols=True, disable=self.hparams["noprogressbar"]
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)

        self.on_fit_end()

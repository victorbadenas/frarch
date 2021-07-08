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
import torch
import logging

from frarch.utils.data import create_dataloader

logger = logging.getLogger(__name__)
PYTHON_VERSION_MAJOR = 3
PYTHON_VERSION_MINOR = 6

default_values = {
    "debug": False,
    "device": "cpu",
    "nonfinite_patience": 3,
    "noprogressbar": False,
    "ckpt_interval_minutes": 0,
}

class ClassifierTrainer:
    def __init__(self, modules, optimizer, hparams, checkpointer=None):
        self.hparams = hparams
        self.optimizer = optimizer
        self.checkpointer = checkpointer

        for name, value in default_values.items():
            if name in self.hparams:
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

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def on_stage_start(self, stage, epoch=None):
        pass

    def on_stage_end(self, stage, epoch=None):
        pass

    def on_train_start(self, stage, epoch=None):
        pass

    def on_train_end(self, stage, epoch=None):
        pass

    def on_validation_start(self, stage, epoch=None):
        pass

    def on_validation_end(self, stage, epoch=None):
        pass

    def on_test_start(self, stage, epoch=None):
        pass

    def on_test_end(self, stage, epoch=None):
        pass

    def on_train_interval(self, stage, epoch=None):
        pass

    def forward(self, batch):
        raise NotImplementedError

    def compute_loss(self, predictions, batch, size):
        raise NotImplementedError

    def fit(self, train_set,
            valid_set=None,
            train_loader_kwargs:dict=None,
            valid_loader_kwargs:dict=None
        ):

        if train_loader_kwargs is None:
            train_loader_kwargs = {}
        if valid_loader_kwargs is None:
            valid_loader_kwargs = {}

        train_dataloader = create_dataloader(train_set, **train_loader_kwargs)
        if valid_set is not None:
            valid_dataloader = create_dataloader(valid_set, **valid_loader_kwargs)

        raise NotImplementedError

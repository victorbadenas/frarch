"""
train_fashion_mnist
*************
:Description: script to train the fashionMNist task
    
:Authors: victor badenas (victor.badenas@gmail.com)
    
:Version: 0.1.0
:Created on: 01/06/2021 11:00 
"""

__title__ = 'train_fashion_mnist'
__version__ = '0.1.0'
__author__ = 'victor badenas'

import os
import sys
import torch
import logging
import frarch as fr
from pprint import pprint
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

logger = logging.getLogger(__name__)

from frarch.utils.stages import Stage

class FMNISTTrainer(fr.train.ClassifierTrainer):
    def forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        embeddings = self.modules.model(inputs)
        return self.modules.classifier(embeddings)

    def compute_loss(self, predictions, batch, stage):
        _, labels = batch
        loss = self.hparams['loss'](predictions, labels)
        if stage == Stage.VALID and 'error_metrics' in self.hparams:
            self.hparams['error_metrics'].update(predictions, labels)
        return loss

    def on_stage_end(self, stage, loss=None, epoch=None):
        if stage == Stage.VALID:
            metrics = self.hparams['error_metrics'].get_metrics(mode="mean")
            logging.info(f'epoch {epoch} validation: {metrics}')


if __name__ == '__main__':
    hparam_file, args = fr.parse_arguments()

    with open(hparam_file, 'r') as hparam_file_handler:
        hparams = load_hyperpyyaml(hparam_file_handler, args, overrides_must_match=False)

    trainer = FMNISTTrainer(
        modules=hparams['modules'],
        opt_class=hparams['opt_class'],
        hparams=hparams,
        checkpointer=None
    )

    trainer.fit(
        train_set=hparams['train_dataset'],
        valid_set=hparams['valid_dataset'],
        train_loader_kwargs=hparams['dataloader_options'],
        valid_loader_kwargs=hparams['dataloader_options']
    )
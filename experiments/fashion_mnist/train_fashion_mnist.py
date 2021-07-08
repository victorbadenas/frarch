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


class FMNISTTrainer(fr.train.ClassifierTrainer):
    pass

if __name__ == '__main__':
    hparam_file, args = fr.parse_arguments()

    with open(hparam_file, 'r') as hparam_file_handler:
        hparams = load_hyperpyyaml(hparam_file_handler, args, overrides_must_match=False)

    pprint(hparams)

    trainer = FMNISTTrainer(
        modules=hparams['modules'],
        optimizer=hparams['optimizer'],
        hparams=hparams,
        checkpointer=None
    )

    trainer.fit(
        train_set=hparams['train_dataset'],
        valid_set=hparams['valid_dataset'],
        train_loader_kwargs=hparams['dataloader_options'],
        valid_loader_kwargs=hparams['dataloader_options']
    )
# Frarch <img src="docs/logo.png" alt="drawing" width="30"/>

![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/victorbadenas/9b54bd086e121233d2ad9a62d2136258/raw/frarch__heads_master.json&style=flat)
![Pytorch](https://img.shields.io/static/v1?label=PyTorch&message=v1.9.1&color=orange&style=flat&logo=pytorch)
![python](https://img.shields.io/pypi/pyversions/frarch?logo=python&style=flat)

![CI](https://github.com/victorbadenas/frarch/actions/workflows/python-app.yml/badge.svg?style=flat)
[![PyPI version fury.io](https://badge.fury.io/py/frarch.svg?style=flat)](https://pypi.python.org/pypi/frarch/)
![license](https://img.shields.io/github/license/victorbadenas/frarch?style=flat)

Frarch is a **Fra**mework for Pyto**rch** experiments inspired by [speechbrain's](https://speechbrain.github.io/) workflow using [hyperpyyaml](https://github.com/speechbrain/HyperPyYAML) configuration files. Frarch aims to minimize the code needed to perform an experiment while organizing the output models and the log files for the experiment as well as the configuration files used to train them in an organised manner.

## Features

- `CPU` and `CUDA` computations. Note that CUDA must be installed for Pytorch and as such frarch to compute in an NVIDIA GPU. Multi-GPU is not supported at the moment, but will be supported in the future.
- Minimize the size of training scripts.
- Support for Python's 3.7, 3.8 and 3.9 versions
- yaml definition of training hyperparameters.
- organisation of output models and their hyperparameters, training scripts and logs.

## Quick installation

The frarch package is evolving and not yet in a stable release. Documentation will be added as the package progresses. The package can be installed via PyPI or via github for the users that what to modify the contents of the package.

### PyPI installation

Once the python environment has been created, you can install frarch by executing:

```bash
pip install frarch
```

Then frarch can be used in a python script using:

```python
import frarch as fr
```

### Github install

Once the python environment has been created, you can install frarch by executing:

```bash
git clone https://github.com/victorbadenas/frarch.git
cd frarch
python setup.py install
```

for development instead of the last command, run `python setup.py develop` to be able to hot reload changes to the package.

### Test

To run the tests for the frarch package:

```bash
python setup.py install
python -m unittest discover
```

### Documentation

To create the documentation, run the following command:

```bash
make -C docs html
sensible-browser docs/_build/html/index.html
make -C docs latexpdf
```

## Running an experiment

Frarch provides training classes such as [`ClassifierTrainer`](https://victorbadenas.github.io/frarch/source/packages/frarch.train.classifier_trainer.html) which provides methods to train a classifier model.

### Example Python trainer script

In this example we present a sample training script for training the MNIST dataset.

```python
from hyperpyyaml import load_hyperpyyaml

import frarch as fr

from frarch.utils.data import build_experiment_structure
from frarch.utils.stages import Stage


class MNISTTrainer(fr.train.ClassifierTrainer):
    def forward(self, batch, stage):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.modules.model(inputs)

    def compute_loss(self, predictions, batch, stage):
        _, labels = batch
        labels = labels.to(self.device)
        return self.hparams["loss"](predictions, labels)

    def on_stage_end(self, stage, loss=None, epoch=None):
        if stage == Stage.VALID:
            if self.checkpointer is not None:
                self.checkpointer.save(epoch=self.current_epoch, current_step=self.step)


if __name__ == "__main__":
    hparam_file, args = fr.parse_arguments()

    with open(hparam_file, "r") as hparam_file_handler:
        hparams = load_hyperpyyaml(
            hparam_file_handler, args, overrides_must_match=False
        )

    build_experiment_structure(
        hparam_file,
        overrides=args,
        experiment_folder=hparams["experiment_folder"],
        debug=hparams["debug"],
    )

    trainer = MNISTTrainer(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    trainer.fit(
        train_set=hparams["train_dataset"],
        valid_set=hparams["valid_dataset"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
```

And the hparams yaml file used to configure the experiment:

```yaml
# seeds
seed: 42
__set_seed: !apply:torch.manual_seed [!ref <seed>]
experiment_name: "mnist"
experiment_folder: "results/mnist_demo/"
device: "cpu"

# data folder
data_folder: /tmp/

# training parameters
epochs: 2
batch_size: 128
shuffle: True
num_clases: 10

transform_tensor: !new:torchvision.transforms.ToTensor
preprocessing: !new:torchvision.transforms.Compose
    transforms: [
        !ref <transform_tensor>,
    ]

# dataset object
train_dataset: !new:torchvision.datasets.MNIST
    root: !ref <data_folder>
    train: true
    download: true
    transform: !ref <preprocessing>

valid_dataset: !new:torchvision.datasets.MNIST
    root: !ref <data_folder>
    train: false
    download: true
    transform: !ref <preprocessing>

# dataloader options
dataloader_options:
    batch_size: !ref <batch_size>
    shuffle: !ref <shuffle>
    num_workers: 8

opt_class: !name:torch.optim.Adam
    lr: 0.001

loss: !new:torch.nn.CrossEntropyLoss

model: !apply:torchvision.models.vgg11
    pretrained: false

modules:
    model: !ref <model>

checkpointer: !new:frarch.modules.Checkpointer
    save_path: !ref <experiment_folder>
    modules: !ref <modules>

```

For the code execution run:

```bash
python train.py mnist.yaml
```

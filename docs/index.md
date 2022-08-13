# Frarch <img src="docs/logo.png" alt="drawing" width="30"/>

![Pytorch](https://img.shields.io/static/v1?label=PyTorch&message=v1.9.1&color=orange&style=flat&logo=pytorch)
![python](https://img.shields.io/pypi/pyversions/frarch?logo=python&style=flat)

![CI](https://github.com/victorbadenas/frarch/actions/workflows/python-app.yml/badge.svg?style=flat)
![docs](https://github.com/victorbadenas/frarch/actions/workflows/docs.yaml/badge.svg?style=flat)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/victorbadenas/9b54bd086e121233d2ad9a62d2136258/raw/frarch__heads_master.json&style=flat)


[![PyPI version fury.io](https://badge.fury.io/py/frarch.svg?style=flat)](https://pypi.python.org/pypi/frarch/)
![license](https://img.shields.io/github/license/victorbadenas/frarch?style=flat)

Frarch is a **Fra**mework for Pyto**rch** experiments inspired by [speechbrain's](https://speechbrain.github.io/) workflow using [hyperpyyaml](https://github.com/speechbrain/HyperPyYAML) configuration files. Frarch aims to minimize the code needed to perform an experiment while organizing the output models and the log files for the experiment as well as the configuration files used to train them in an organised manner.

## Features

- `CPU` and `CUDA` computations. Note that CUDA must be installed for Pytorch and as such frarch to compute in an NVIDIA GPU. Multi-GPU is not supported at the moment, but will be supported in the future.
- Minimize the size of training scripts.
- Support for Python's 3.8 and 3.9 versions
- yaml definition of training hyperparameters.
- organisation of output models and their hyperparameters, training scripts and logs.

## Documentation index

- [Installation](docs/installation.md)
- [Running an experiment](docs/running_an_experiment.md)
- [Documentation](docs/documentation.md)
- [Testing](docs/testing.md)

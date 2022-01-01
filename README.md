# Frarch <img src="docs/logo.png" alt="drawing" width="30"/>

![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/vbadenas/9b54bd086e121233d2ad9a62d2136258/raw/frarch__heads_master.json&style=flat)
![Pytorch](https://img.shields.io/static/v1?label=PyTorch&message=v1.9.1&color=orange&style=flat&logo=pytorch)
![python](https://img.shields.io/pypi/pyversions/frarch?logo=python&style=flat)

![CI](https://github.com/vbadenas/frarch/actions/workflows/python-app.yml/badge.svg?style=flat)
[![PyPI version fury.io](https://badge.fury.io/py/frarch.svg?style=flat)](https://pypi.python.org/pypi/frarch/)
![license](https://img.shields.io/github/license/vbadenas/frarch?style=flat)

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
git clone https://github.com/vbadenas/frarch.git
cd frarch
python setup.py install
```

for development instead of the last command, run `python setup.py develop` to be able to hot reload changes to the package.

### Tests

To run the tests for the frarch package:

```bash
python setup.py install
python -m unittest discover
```

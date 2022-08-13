# Installation

## Quick installation

The frarch package is evolving and not yet in a stable release. Documentation will be added as the package progresses. The package can be installed via PyPI or via github for the users that what to modify the contents of the package.

## PyPI installation

Once the python environment has been created, you can install frarch by executing:

```bash
pip install frarch
```

Then frarch can be used in a python script using:

```python
import frarch as fr
```

## Github install

Once the python environment has been created, you can install frarch by executing:

```bash
git clone https://github.com/victorbadenas/frarch.git
cd frarch
pip install . # for enabling editable mode use the `-e` flag
```

for development instead of the last command, run `pip install -e .[dev]` to be able to hot reload changes to the package.

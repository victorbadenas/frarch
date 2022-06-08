from importlib.metadata import metadata
from pathlib import Path

from . import datasets
from . import models
from . import modules
from . import train
from . import utils
from .parser import parse_arguments

__meta__ = type("FrarchMeta", (), metadata("frarch").json)

__version__ = __meta__.version

print()

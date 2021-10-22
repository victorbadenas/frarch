from . import __meta__

__version__ = __meta__.version

from . import datasets, models, modules, train, utils
from .parser import parse_arguments

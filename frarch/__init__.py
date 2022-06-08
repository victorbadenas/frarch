from pathlib import Path

from . import datasets
from . import models
from . import modules
from . import train
from . import utils
from .parser import parse_arguments

__version__ = Path(__file__).with_name("_version.txt").read_text("utf-8").strip()

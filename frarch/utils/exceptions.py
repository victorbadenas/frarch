from pathlib import Path
from typing import Union


class FrarchException(Exception):
    """Base exception class for frarch package."""


class DatasetNotFoundError(FrarchException):
    """Exception raised for OS dataset errors.

    Args:
        path ([Path, str]): path where the dataset should be
    """

    def __init__(
        self, path: Union[str, Path], msg: str = "Dataset not found in path {path}"
    ) -> None:
        self.path = path
        self.msg = msg
        super().__init__(self.msg.format(path=path))

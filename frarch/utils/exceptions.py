from pathlib import Path
from typing import Union


class DatasetNotFoundError(Exception):
    """Exception raised for OS dataset errors.

    Args
    ----
        path([Path, str]): [path where the dataset should be]
    """

    def __init__(self, path: Union[str, Path], msg="Dataset not found in path {path}"):
        self.path = path
        self.msg = msg
        super().__init__(self.msg.format(path=path))

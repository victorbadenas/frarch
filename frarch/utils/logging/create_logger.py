import logging
import sys
from pathlib import Path
from typing import Union


def create_logger_file(log_file_path: Union[str, Path], debug=False, stdout=False):
    if not isinstance(log_file_path, (str, Path)):
        raise ValueError("path must be a string or Path object")
    if not isinstance(debug, bool):
        raise ValueError("debug must be boolean")
    if not isinstance(stdout, bool):
        raise ValueError("stdout must be boolean")

    # cast to path
    log_file_path = Path(log_file_path)

    # get logging level
    level = logging.DEBUG if debug else logging.INFO

    # set logging format
    logging_format = (
        "[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]"
        "-%(levelname)s: %(message)s"
    )

    # cretate parent and create file handler for config
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)

    # if stdout add console_handler
    if stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(logging_format))
        logging.getLogger().addHandler(console_handler)

import logging
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Union
from urllib.request import urlretrieve

import torch
from hyperpyyaml import resolve_references
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from frarch.utils.logging.create_logger import create_logger_file

logger = logging.getLogger(__name__)


def create_dataloader(dataset: Dataset, **dataloader_kwargs) -> DataLoader:
    """Create dataloader from dataset.

    Args:
        dataset (torch.utils.data.Dataset): dataset object to feed onto DataLoader.

    Raises:
        ValueError: dataset is not a Dataset or does not inherit from it.

    Returns:
        torch.utils.data.DataLoader: dataloader with the dataset object.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset needs to be a child or torch.utils.data.Dataset")
    return DataLoader(dataset, **dataloader_kwargs)


def build_experiment_structure(
    hparams_file: Union[str, Path],
    experiment_folder: Union[str, Path],
    overrides: Mapping = None,
    debug: bool = False,
):
    """Construct experiment folder hierarchy on experiment_folder.

    Args:
        hparams_file (Union[str, Path]): hparams configuration file path.
        experiment_folder (Union[str, Path]): Folder where to store experiment files.
            Defaults to "results/debug/".
        overrides (Mapping, optional): Parameters to override on hparams file.
            Defaults to None.
        debug (bool, optional): debug flag. Defaults to False.
    """
    if overrides is None:
        overrides = {}

    base_path = Path(experiment_folder)
    base_path.mkdir(exist_ok=True, parents=True)
    save_path = base_path / "save"
    save_path.mkdir(exist_ok=True, parents=True)

    experiment_yaml_path = base_path / "train.yaml"
    with open(hparams_file, "r") as f:
        hparams = resolve_references(f, overrides).getvalue()
    with open(experiment_yaml_path, "w") as f:
        f.write(hparams)

    log_file_path = base_path / "train.log"
    create_logger_file(log_file_path, debug=debug, stdout=debug)

    logger.info(f"experiment folder {str(base_path)} created successfully")


def download_url(
    url: str, destination: Optional[str] = None, progress_bar: bool = True
) -> str:
    """Download a URL to a local file.

    Args:
        url (str): The URL to download.
        destination (Optional[str], optional): The destination of the file. If None is
            given the file is saved to a temporary directory. Defaults to None.
        progress_bar (bool, optional): The destination of the file. If None is given
            the file is saved to a temporary directory. Defaults to True.

    Returns:
        str: filename of downloaded file.
    """

    def update_progressbar(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
            filename, _ = urlretrieve(
                url, filename=destination, reporthook=update_progressbar(t)
            )
    else:
        filename, _ = urlretrieve(url, filename=destination)
    return filename


def tensor_in_device(data: Any, device: str = "cpu", **kwargs) -> torch.Tensor:
    """Create tensor in device.

    Args:
        data (Any): data on the tensor.
        device (str, optional): string of the device to be created in.
            Defaults to "cpu".

    Returns:
        torch.Tensor: tensor in device.
    """
    return torch.Tensor(data, **kwargs).to(device)


def read_file(filepath: Union[str, Path]) -> str:
    """Read contents of file.

    Args:
        filepath (Union[str, Path]): path to file.

    Returns:
        str: contents of the file
    """
    return Path(filepath).read_text()

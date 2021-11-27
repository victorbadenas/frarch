import logging
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

import torch
from hyperpyyaml import resolve_references
from tqdm import tqdm

from frarch.utils.logging import create_logger_file

logger = logging.getLogger(__name__)


def create_dataloader(dataset: torch.utils.data.Dataset, **dataloader_kwargs):
    if not isinstance(dataset, torch.utils.data.Dataset):
        raise ValueError("dataset needs to be a child or torch.utils.data.Dataset")
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def build_experiment_structure(
    hparams_file,
    overrides=None,
    experiment_folder: str = "results/debug/",
    debug: bool = False,
):
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


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to
        a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation:
        https://github.com/tqdm/tqdm
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


def tensorInDevice(data, device="cpu", **kwargs):
    return torch.Tensor(data, **kwargs).to(device)


def read_file(filepath: Union[str, Path]):
    return Path(filepath).read_text()

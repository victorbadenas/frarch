from pathlib import Path
import logging
import hyperpyyaml
import torch
from hyperpyyaml import resolve_references

from frarch.utils.logging import create_logger

logger = logging.getLogger(__name__)

def create_dataloader(dataset: torch.utils.data.Dataset, **dataloader_kwargs):
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def build_experiment_structure(
    hparams_file, overrides=None, experiment_folder: str = "results/debug/", debug: bool = False
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
    create_logger(log_file_path, debug=debug, stdout=debug)

    logger.info(f"experiment folder {str(base_path)} created successfully")

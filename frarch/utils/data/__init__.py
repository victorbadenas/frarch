from pathlib import Path

import hyperpyyaml
import torch
from hyperpyyaml import resolve_references

from frarch.utils.logging import create_logger


def create_dataloader(dataset: torch.utils.data.Dataset, **dataloader_kwargs):
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def build_experiment_structure(
    hparams_file, overrides=None, experiment_name: str = "debug", debug: bool = False
):
    if overrides is None:
        overrides = {}

    base_path = Path("results") / experiment_name
    base_path.mkdir(exist_ok=True, parents=True)
    experiment_yaml_path = base_path / "train.yaml"
    with open(hparams_file, "r") as f:
        hparams = resolve_references(f, overrides).getvalue()
    with open(experiment_yaml_path, "w") as f:
        f.write(hparams)
    log_file_path = base_path / "train.log"
    create_logger(log_file_path, debug=debug, stdout=debug)

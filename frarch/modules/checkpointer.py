import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Union

import torch

logger = logging.getLogger(__name__)

METRIC_MODES = ["min", "max"]
LOAD_MODES = ["last", "best"]


class Checkpointer:
    """Class for managing checkpoints.

    Args:
        save_path (Union[str, Path]): folder to store the checkpoint and training
            data to.
        modules (Mapping[str, torch.nn.Module]): dict-like structure with modules.
        save_best_only (bool, optional): If true, save only the best model according to
            some metric. If True, reference metric should be specified. If False, save
            all end of epoch checkpoints. Defaults to False.
        reference_metric (str, optional): Metric to use to determine the best model when
            save_best_only is True. Must be a string in the keys of the modules
            dict-like structure. Defaults to None.
        mode (str, optional): min if lower is better, max if higher is better.
            Examples: min for error and max for accuracy. Defaults to "min".

    Raises:
        ValueError: modules are not a dict or torch.nn.ModuleDict instance
        ValueError: modules in modules dict-like don't have string keys or
            torch.nn.Module values
            ValueError: path must be a string or Path object
        ValueError: metadata key is reserved for the metadata.json object.
        ValueError: metric mode is not min or max
        ValueError: save_best_only is True and no metric is defined.
    """

    def __init__(
        self,
        save_path: Union[str, Path],
        modules: torch.nn.ModuleDict,
        save_best_only: bool = False,
        reference_metric: str = None,
        mode: str = "min",
    ) -> None:
        if not isinstance(modules, (dict, torch.nn.ModuleDict)):
            raise ValueError("modules must be a dict or torch.nn.ModuleDict instance")
        elif not all(
            isinstance(k, str) and isinstance(v, torch.nn.Module)
            for k, v in modules.items()
        ):
            raise ValueError("modules must have string keys and torch.nn.Module values")
        if not isinstance(save_path, (str, Path)):
            raise ValueError("path must be a string or Path object")

        if "metadata" in modules:
            raise ValueError("metadata in modules is reserved for metadata json object")
        if mode not in METRIC_MODES:
            raise ValueError(f"metric mode must be in {METRIC_MODES} not {mode}")
        if save_best_only and reference_metric is None:
            raise ValueError(
                "specify the metric name to consider while save_best_only=True"
            )

        self.base_path = Path(save_path) / "save"
        self.metadata_file_name = "metadata.json"
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.modules = modules
        self.metadata = {}
        self.save_best_only = save_best_only
        self.best_metric = None
        self.reference_metric = reference_metric
        self.mode = mode

    def save_initial_weights(self) -> None:
        """Save weights with which the model has been initialized."""
        time_str = str(datetime.now())
        ckpt_folder = "initial_weights"
        paths = self._build_paths(ckpt_folder)
        self._save_modules(paths)
        self._save_metadata(
            time_str,
            paths["metadata"],
            epoch=-1,
            intra_epoch=False,
            step=0,
        )

    def save(
        self,
        epoch: int,
        current_step: int,
        intra_epoch: bool = False,
        extra_data: Dict = None,
        **metrics: Any,
    ) -> None:
        """Save checkpoint.

        Args:
            epoch (int): current epoch index.
            current_step (int): current batch index.
            intra_epoch (bool, optional): boolean flag to indicate if the checkpoint is
                intra epoch if true and end of epoch if false. Defaults to False.
            extra_data (Dict, optional): extra metadata in json format to add to the
                metadata.json file. Defaults to None.
        """
        time_str = str(datetime.now())
        ckpt_folder = f"ckpt_{time_str.replace(' ', '_')}"

        paths = self._build_paths(ckpt_folder)
        self._save_modules(paths)
        self._save_metadata(
            time_str,
            paths["metadata"],
            epoch=epoch,
            intra_epoch=intra_epoch,
            step=current_step,
            extra_data=extra_data,
            **metrics,
        )
        if intra_epoch:
            logger.info(f"Saved intra_epoch model to {ckpt_folder}")
        else:
            logger.info(f"Saved end_of_epoch model to {ckpt_folder}")

        if not intra_epoch and self.reference_metric is not None:
            self._update_best_metric(**metrics)
            if self.save_best_only:
                self._remove_old_ckpts(ckpt_folder)

    def load(self, mode="last", **load_kwargs) -> bool:
        """Load checkpoint from folder.

        Args:
            mode (str, optional): last for loading the last checkpoint stored and best
                to load the model with the mest metric. Defaults to "last".

        Raises:
            ValueError: mode is not best or last
        """
        if mode == "best":
            return self._load_best_checkpoint(**load_kwargs)
        elif mode == "last":
            return self._load_last_checkpoint(**load_kwargs)
        else:
            raise ValueError('load\'s mode kwarg can be "best" or "last"')

    def exists_checkpoint(self) -> bool:
        """Check if save_path contains a checkpoint folder.

        Returns:
            bool: True if it contains a checkpoint folder, False if not.
        """
        for folder in self.base_path.iterdir():
            if self._is_ckpt_dir(folder):
                return True
        return False

    @property
    def current_epoch(self) -> int:
        if len(self.metadata) > 0 and "epoch" in self.metadata:
            return int(self.metadata["epoch"])
        return 0

    @property
    def next_epoch(self) -> int:
        if self._is_intraepoch():
            return self.current_epoch
        else:
            return self.current_epoch + 1

    @property
    def step(self) -> int:
        if self._is_intraepoch():
            return self.metadata["step"]
        return 0

    def _build_paths(self, ckpt_folder_name: str) -> dict:
        paths = {}
        for module_name in self.modules.keys():
            module_path = self.base_path / ckpt_folder_name / f"{module_name}.pt"
            paths[module_name] = module_path
        paths["metadata"] = self.base_path / ckpt_folder_name / self.metadata_file_name
        return paths

    def _save_modules(self, paths: Dict[str, Path]):
        for module_name in self.modules:
            paths[module_name].parent.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.modules[module_name].state_dict(),
                paths[module_name],
            )

    def _save_metadata(
        self,
        time_str: str,
        metadata_path: Union[Path, str],
        epoch: int,
        step: int,
        intra_epoch: bool = False,
        extra_data: Dict = None,
        **metrics,
    ) -> None:
        self.metadata = {
            "intra_epoch": intra_epoch,
            "step": step,
            "epoch": epoch,
            "time": time_str,
            "extra_info": extra_data,
            **metrics,
        }

        with open(metadata_path, "w") as metadata_handler:
            json.dump(self.metadata, metadata_handler, indent=4)

    def _update_best_metric(self, **metrics):
        if self.best_metric is None:
            self.best_metric = metrics[self.reference_metric]
        else:
            if self._is_better(metrics[self.reference_metric], self.best_metric):
                self.best_metric = metrics[self.reference_metric]

    def _remove_old_ckpts(self, curr_ckpt_folder):
        for old_ckpt in self.base_path.iterdir():
            if old_ckpt.name == curr_ckpt_folder or not str(old_ckpt.name).startswith(
                "ckpt_"
            ):
                continue
            with open(old_ckpt / self.metadata_file_name, "r") as metadata_handler:
                old_metadata = json.load(metadata_handler)

            if self.reference_metric not in old_metadata or not self._is_better(
                old_metadata[self.reference_metric], self.best_metric
            ):
                shutil.rmtree(old_ckpt)

    def _is_better(self, new_metric, old_metric) -> bool:
        if self.mode == "min":
            return new_metric <= old_metric
        elif self.mode == "max":
            return new_metric >= old_metric

    def _load_best_checkpoint(self, **load_kwargs) -> bool:
        ckpts_meta = self._load_checkpoints_meta()
        ckpts_meta.pop("initial_weights")
        cmp_fn = min if self.mode == "min" else max
        best_ckpt_name = cmp_fn(
            ckpts_meta, key=lambda i: ckpts_meta[i][self.reference_metric]
        )
        return self._load_checkpoint_from_folder(
            best_ckpt_name, ckpts_meta[best_ckpt_name], **load_kwargs
        )

    def _load_last_checkpoint(self, **load_kwargs) -> bool:
        ckpts_meta = self._load_checkpoints_meta()
        latest_ckpt_name = max(ckpts_meta, key=lambda i: ckpts_meta[i]["time"])
        return self._load_checkpoint_from_folder(
            latest_ckpt_name, ckpts_meta[latest_ckpt_name], **load_kwargs
        )

    def _load_checkpoints_meta(self) -> dict:
        ckpts_meta = {}
        for folder in self.base_path.iterdir():
            if not folder.is_dir():
                continue

            if self._is_ckpt_dir(folder):
                metadata_path = folder / self.metadata_file_name
                with open(metadata_path, "r") as f:
                    ckpts_meta[folder.name] = json.load(f)

                ckpts_meta[folder.name]["time"] = datetime.strptime(
                    ckpts_meta[folder.name]["time"], "%Y-%m-%d %H:%M:%S.%f"
                )

        return ckpts_meta

    def _load_checkpoint_from_folder(
        self, ckpt_folder_name, metadata, **load_kwargs
    ) -> bool:
        paths = self._build_paths(ckpt_folder_name)
        for module_name in self.modules:
            try:
                self.modules[module_name].load_state_dict(
                    torch.load(paths[module_name], **load_kwargs)
                )
            except Exception as e:
                logger.error(f"Failed loading ckpt from {ckpt_folder_name}.")
                logger.error(e)
                raise e
        self.metadata = metadata
        logger.info(
            f"Loaded ckpt from epoch {self.current_epoch} from {ckpt_folder_name}"
        )
        return True

    @staticmethod
    def _is_ckpt_dir(path: Union[str, Path]):
        return str(path.name).startswith("ckpt_") or str(path.name) == "initial_weights"

    def _is_intraepoch(self) -> bool:
        return self.metadata["intra_epoch"]

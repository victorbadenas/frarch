import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch

logger = logging.getLogger(__name__)

METRIC_MODES = ["min", "max"]
LOAD_MODES = ["last", "best"]


class Checkpointer:
    def __init__(
        self,
        save_path: Union[str, Path],
        modules: Mapping[str, torch.nn.Module],
        save_best_only: bool = True,
        reference_metric: str = None,
        mode: str = "min",
    ):
        if not isinstance(modules, (dict, torch.nn.ModuleDict)):
            raise ValueError("modules must be a dict or torch.nn.ModuleDict instance")
        elif not all(isinstance(k, str) and isinstance(v, torch.nn.Module) for k, v in modules.items()):
            raise ValueError("modules must have string keys and torch.nn.Module values")
        if not isinstance(save_path, (str, Path)):
            raise ValueError("path must be a string or Path object")

        if "metadata" in modules:
            raise ValueError(
                "metadata in modules is reserved for metadata json object"
            )
        if mode not in METRIC_MODES:
            raise ValueError(f"metric mode must be in {METRIC_MODES} not {mode}")
        if save_best_only and reference_metric is None:
            raise ValueError(
                "specify the metric name to consider while save_best_only=True"
            )

        self.base_path = Path(save_path) / "save"
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.modules = modules
        self.metadata = {}
        self.save_best_only = save_best_only
        self.best_metric = None
        self.reference_metric = reference_metric
        self.mode = mode

    def save_initial_weights(self):
        time_str = str(datetime.now())
        ckpt_folder = "initial_weights"
        paths = self._build_paths(ckpt_folder)
        self._save_modules(paths)
        self._save_json(
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
        **metrics: Dict[str, Any],
    ):
        time_str = str(datetime.now())
        ckpt_folder = f"ckpt_{time_str.replace(' ', '_')}"

        paths = self._build_paths(ckpt_folder)
        self._save_modules(paths)
        self._save_json(
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
            self.update_best_metric(**metrics)
            if self.save_best_only:
                self.remove_old_ckpts(ckpt_folder)

    def _build_paths(self, ckpt_folder_name: str) -> dict:
        paths = {}
        for module_name in self.modules.keys():
            module_path = self.base_path / ckpt_folder_name / f"{module_name}.pt"
            paths[module_name] = module_path
        paths["metadata"] = self.base_path / ckpt_folder_name / "metadata.json"
        return paths

    def _save_modules(self, paths: Dict[str, Path]):
        for module_name in self.modules:
            paths[module_name].parent.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.modules[module_name].state_dict(),
                paths[module_name],
            )

    def _save_json(
        self,
        time_str: str,
        metadata_path: Union[Path, str],
        epoch: int,
        step: int,
        intra_epoch: bool = False,
        extra_data: Dict = None,
        **metrics,
    ):
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

    def update_best_metric(self, **metrics):
        if self.best_metric is None:
            self.best_metric = metrics[self.reference_metric]
        else:
            if self.is_better(metrics[self.reference_metric], self.best_metric):
                self.best_metric = metrics[self.reference_metric]

    def remove_old_ckpts(self, curr_ckpt_folder):
        for old_ckpt in self.base_path.iterdir():
            if old_ckpt.name == curr_ckpt_folder or not str(old_ckpt.name).startswith(
                "ckpt_"
            ):
                continue
            with open(old_ckpt / "metadata.json", "r") as metadata_handler:
                old_metadata = json.load(metadata_handler)

            if self.reference_metric not in old_metadata:
                shutil.rmtree(old_ckpt)
            elif not self.is_better(
                old_metadata[self.reference_metric], self.best_metric
            ):
                shutil.rmtree(old_ckpt)

    def is_better(self, new_metric, old_metric) -> bool:
        if self.mode == "min":
            return new_metric <= old_metric
        elif self.mode == "max":
            return new_metric >= old_metric

    def load(self, mode="last") -> bool:
        if mode == "best":
            return self.load_best_checkpoint()
        elif mode == "last":
            return self.load_last_checkpoint()
        else:
            raise ValueError('load\'s mode kwarg can be "best" or "last"')

    def exists_checkpoint(self) -> bool:
        for folder in self.base_path.iterdir():
            if self.is_ckpt_dir(folder):
                return True
        return False

    def load_best_checkpoint(self) -> bool:
        ckpts_meta = self.load_checkpoints_meta()
        ckpts_meta.pop("initial_weights")
        cmp_fn = min if self.mode == "min" else max
        best_ckpt_name = cmp_fn(
            ckpts_meta, key=lambda i: ckpts_meta[i][self.reference_metric]
        )
        return self.load_checkpoint_from_folder(
            best_ckpt_name, ckpts_meta[best_ckpt_name]
        )

    def load_last_checkpoint(self) -> bool:
        ckpts_meta = self.load_checkpoints_meta()
        latest_ckpt_name = max(ckpts_meta, key=lambda i: ckpts_meta[i]["time"])
        return self.load_checkpoint_from_folder(
            latest_ckpt_name, ckpts_meta[latest_ckpt_name]
        )

    def load_checkpoints_meta(self) -> dict:
        ckpts_meta = {}
        for folder in self.base_path.iterdir():
            if not folder.is_dir():
                continue

            if self.is_ckpt_dir(folder):
                metadata_path = folder / "metadata.json"
                with open(metadata_path, "r") as f:
                    ckpts_meta[folder.name] = json.load(f)

                ckpts_meta[folder.name]["time"] = datetime.strptime(
                    ckpts_meta[folder.name]["time"], "%Y-%m-%d %H:%M:%S.%f"
                )

        return ckpts_meta

    def load_checkpoint_from_folder(self, ckpt_folder_name, metadata) -> bool:
        paths = self._build_paths(ckpt_folder_name)
        for module_name in self.modules:
            try:
                self.modules[module_name].load_state_dict(
                    torch.load(
                        paths[module_name],
                    )
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
    def is_ckpt_dir(path: Union[str, Path]):
        return str(path.name).startswith("ckpt_") or str(path.name) == "initial_weights"

    def is_intraepoch(self) -> bool:
        return self.metadata["intra_epoch"]

    @property
    def current_epoch(self) -> int:
        if len(self.metadata) >= 0 and "epoch" in self.metadata:
            return int(self.metadata["epoch"])
        return 0

    @property
    def next_epoch(self) -> int:
        if self.is_intraepoch():
            return self.current_epoch
        else:
            return self.current_epoch + 1

    @property
    def step(self) -> int:
        if self.is_intraepoch():
            return self.metadata["step"]
        return 0

import torch
import json
import logging
from datetime import datetime
import shutil
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

METRIC_MODES = ["min", "max"]
LOAD_MODES = ["last", "best"]

class Checkpointer:
    def __init__(self, save_path, modules, save_best_only:bool=True, reference_metric:str=None, mode:str="min"):
        if "metadata" in modules:
            raise ValueError("metadata in moddules is reserved for metadata json object")
        if not mode in METRIC_MODES:
            raise ValueError(f"metric mode must be in {METRIC_MODES} not {mode}")
        if save_best_only and reference_metric is None:
            raise ValueError("specify the metric name to consider while save_best_only=True")

        self.modules = modules
        self.base_path = Path(save_path) / "save"
        self.metadata = {}
        self.save_best_only = save_best_only
        self.best_metric = None
        self.reference_metric = reference_metric
        self.mode = mode

    def build_paths(self, ckpt_folder_name:str) -> dict:
        paths = {}
        for module_name, module in self.modules.items():
            module_path = self.base_path / ckpt_folder_name / f"{module_name}.pt"
            paths[module_name] = module_path
        paths["metadata"] = self.base_path / ckpt_folder_name / "metadata.json"
        return paths

    def save(self, epoch:int, current_step:int, intra_epoch:bool=False, **metrics):
        time_str = str(datetime.now())
        ckpt_folder = f"ckpt_{time_str.replace(' ', '_')}"
        paths = self.build_paths(ckpt_folder)

        for module_name in self.modules:
            paths[module_name].parent.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.modules[module_name].state_dict(), 
                paths[module_name],
            )

        self.save_json(time_str, paths["metadata"], epoch=epoch, intra_epoch=intra_epoch, step=current_step, **metrics)
        self.update_best_metric(**metrics)

        if not intra_epoch and self.save_best_only:
            self.remove_old_ckpts(ckpt_folder)

    def save_json(self, time_str:str, metadata_path:Union[Path, str], epoch:int, step:int, intra_epoch:bool=False, **metrics):
        self.metadata = {
            "intra_epoch": intra_epoch,
            "step": step,
            "epoch": epoch,
            "time": time_str,
            **metrics
        }
        with open(metadata_path, 'w') as metadata_handler:
            json.dump(self.metadata, metadata_handler, indent=4)

    def update_best_metric(self, **metrics):
        if self.best_metric is None:
            self.best_metric = metrics[self.reference_metric]
        else:
            if self.is_better(metrics[self.reference_metric], self.best_metric):
                self.best_metric = metrics[self.reference_metric]

    def remove_old_ckpts(self, curr_ckpt_folder):
        for old_ckpt in self.base_path.iterdir():
            if old_ckpt.name == curr_ckpt_folder or not str(old_ckpt.name).startswith('ckpt_'):
                continue
            with open(old_ckpt / 'metadata.json', 'r') as metadata_handler:
                old_metadata = json.load(metadata_handler)
            if not self.is_better(old_metadata["classification_error"], self.best_metric):
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
            raise ValueError("load's mode kwarg can be \"best\" or \"last\"")

    def exists_checkpoint(self) -> bool:
        for folder in self.base_path.iterdir():
            if str(folder.name).startswith('ckpt_'):
                return True
        return False

    def load_best_checkpoint(self) -> bool:
        ckpts_meta = self.load_checkpoints_meta()
        cmp_fn = min if self.mode == "min" else max
        best_ckpt_name = cmp_fn(ckpts_meta, key=lambda i: ckpts_meta[i][self.reference_metric])
        return self.load_checkpoint_from_folder(best_ckpt_name, ckpts_meta[best_ckpt_name])

    def load_last_checkpoint(self) -> bool:
        ckpts_meta = self.load_checkpoints_meta()
        latest_ckpt_name = max(ckpts_meta, key=lambda i: ckpts_meta[i]['time'])
        return self.load_checkpoint_from_folder(latest_ckpt_name, ckpts_meta[latest_ckpt_name])

    def load_checkpoints_meta(self) -> dict:
        ckpts_meta = {}
        for folder in self.base_path.iterdir():
            if not str(folder.name).startswith('ckpt_') or not folder.is_dir():
                continue

            metadata_path = folder / "metadata.json"
            with open(metadata_path, 'r') as f:
                ckpts_meta[folder.name] = json.load(f)

            ckpts_meta[folder.name]['time'] = datetime.strptime(
                ckpts_meta[folder.name]['time'],
                "%Y-%m-%d %H:%M:%S.%f"
            )
        return ckpts_meta

    def load_checkpoint_from_folder(self, ckpt_folder_name, metadata) -> bool:
        paths = self.build_paths(ckpt_folder_name)
        for module_name in self.modules:
            try:
                self.modules[module_name].load_state_dict(
                    torch.load(
                        paths[module_name],
                    )
                )
            except Exception as e:
                raise e
                # return false in case we want to keep with the training and just warn the user.
                return False
        self.metadata = metadata
        return True

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

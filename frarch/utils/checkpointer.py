import torch
import json
from datetime import datetime
import shutil
from pathlib import Path


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

    def build_paths(self, ckpt_folder_name:str):
        paths = {}
        for module_name, module in self.modules.items():
            module_path = self.base_path / ckpt_folder_name / f"{module_name}.pt"
            paths[module_name] = module_path
        paths["metadata"] = self.base_path / ckpt_folder_name / "metadata.json"
        return paths

    def save(self, epoch=None, **metrics):
        time_str = str(datetime.now())
        ckpt_folder = f"ckpt_{time_str.replace(' ', '_')}"
        paths = self.build_paths(ckpt_folder)

        for module_name in self.modules:
            paths[module_name].parent.mkdir(exist_ok=True, parents=True)
            torch.save(
                self.modules[module_name].state_dict(), 
                paths[module_name],
            )

        self.save_json(time_str, paths["metadata"], epoch=epoch, **metrics)
        self.update_best_metric(**metrics)

        if epoch is not None and self.save_best_only:
            self.remove_old_ckpts(ckpt_folder)

    def save_json(self, time_str:str, metadata_path, epoch=None, **metrics):
        self.metadata = {
            "epoch": epoch if epoch is not None else "intra_epoch_ckpt",
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

    def is_better(self, new_metric, old_metric):
        if self.mode == "min":
            return new_metric <= old_metric
        elif self.mode == "max":
            return new_metric >= old_metric

    def load(self, mode="last"):
        if mode == "best":
            self.load_best_checkpoint()
        elif mode == "last":
            self.load_last_checkpoint()
        else:
            raise ValueError("load's mode kwarg can be \"best\" or \"last\"")

    def exists_checkpoint(self):
        for folder in self.base_path.iterdir():
            if str(folder).startswith('ckpt_'):
                return True
        return False

    def load_best_checkpoint(self):
        pass

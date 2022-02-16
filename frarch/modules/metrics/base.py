import abc

import torch

AGGREGATION_MODES = ["mean", "max", "min"]


class Metric(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.metrics = []

    @abc.abstractmethod
    def update(self, predictions: torch.Tensor, truth: torch.Tensor) -> None:
        pass

    def __len__(self) -> int:
        return len(self.metrics)

    def get_metric(self, mode="mean") -> float:
        if len(self) == 0:
            return 0.0

        if mode not in AGGREGATION_MODES:
            raise ValueError(
                f"Mode {mode} not supported. Supported modes: {AGGREGATION_MODES}"
            )
        if mode == "mean":
            return sum(self.metrics) / len(self)
        elif mode == "max":
            return max(self.metrics)
        elif mode == "min":
            return min(self.metrics)

import abc
from typing import Any

import torch

AGGREGATION_MODES = ["mean", "max", "min"]


class Metric(metaclass=abc.ABCMeta):
    """abstract class for Metric objects.

    Example:
        Simple usage of the Metric class::
            class MyMetric(Metric):
                def _update(self, predictions, truth):
                    # compute some metric
                    return metric_value

            model = MyModel()
            mymetric = MyMetric()
            for batch, labels in dataset:
                predictions = model(batch)
                mymetric.update(predictions, labels)
            print(mymetric.get_metric(mode="mean"))

    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear metrics from class."""
        self.metrics = []

    def update(self, predictions: torch.Tensor, truth: torch.Tensor) -> None:
        """Compute metric value and append to the metrics array.

        Args:
            predictions (torch.Tensor): output tensors from model.
            truth (torch.Tensor): ground truth tensor.
        """
        self.metrics.append(self._update(predictions, truth))

    @abc.abstractmethod
    def _update(self, predictions: torch.Tensor, truth: torch.Tensor) -> Any:
        """Compute the metric value.

        Args:
            predictions (torch.Tensor): output tensors from model.
            truth (torch.Tensor): ground truth tensor.
        """

    def __len__(self) -> int:
        return len(self.metrics)

    def get_metric(self, mode="mean") -> float:
        """Aggregate all values stored in the metric class.

        Args:
            mode (str, optional): aggregation type. mean, max or min.
                Defaults to "mean".

        Raises:
            ValueError: aggregation mode not supported

        Returns:
            float: aggregated metric.
        """
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

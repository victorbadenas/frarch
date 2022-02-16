from typing import Any, Mapping

from .base import Metric


class MetricsWrapper:
    """Store a set of metrics and perform operations in all of them simultaneously."""

    def __init__(self, **kwargs: Metric) -> None:
        for k, v in kwargs.items():
            if not isinstance(v, Metric):
                raise ValueError(f"value for key {k} should inherit from Metric")
            setattr(self, k, v)

    def reset(self) -> None:
        """Call reset in all metrics in the MetricsWrapper class."""
        for _, v in self.__dict__.items():
            v.reset()

    def update(self, *args: Any, **kwargs: Any) -> None:
        for _, v in self.__dict__.items():
            v.update(*args, **kwargs)

    def get_metrics(self, *args: Any, **kwargs: Any) -> Mapping[str, Metric]:
        metrics = {}
        for k, v in self.__dict__.items():
            metrics[k] = v.get_metric(*args, **kwargs)
        return metrics

from typing import Any
from typing import Dict

from .base import Metric


class MetricsWrapper:
    """Store a set of metrics and perform operations in all of them simultaneously.

    Example:
        Sample code for metrics wrapper:
            metrics_wrapper = MetricsWrapper(
                metric_str0=Metric0(),
                metric_str1=Metric1(),
            )
            model = Model()

            for batch, labels in dataset:
                predictions = model(batch)
                metrics_wrapper.update(predictions, labels)
            print(metrics_wrapper.get_metrics())
            # prints {"metric_str0": 0.0, "metric_str1": 1.0}
    """

    def __init__(self, **kwargs: Metric) -> None:
        """Initialize metrics in wrapper.

        Raises:
            ValueError: if any of the values in kwargs don't inherit from metric.
        """
        for k, v in kwargs.items():
            if not isinstance(v, Metric):
                raise ValueError(f"value for key {k} should inherit from Metric")
            setattr(self, k, v)

    def reset(self) -> None:
        """Call reset in all metrics in the MetricsWrapper class."""
        for _, v in self.__dict__.items():
            v.reset()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Call update on all metrics in MetricsWrapper class."""
        for _, v in self.__dict__.items():
            v.update(*args, **kwargs)

    def get_metrics(self, *args: Any, **kwargs: Any) -> Dict[str, Metric]:
        """Build a dict with aggregated metrics.

        Returns:
            Dict[str, Metric]: dict with metric names as keys and aggregated metrics\
                as values.
        """
        metrics = {}
        for k, v in self.__dict__.items():
            metrics[k] = v.get_metric(*args, **kwargs)
        return metrics

from .base import Metric


class MetricsWrapper:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, Metric):
                raise ValueError(f"value for key {k} should inherit from Metric")
            setattr(self, k, v)

    def reset(self):
        for _, v in self.__dict__.items():
            v.reset()

    def update(self, *args, **kwargs):
        for _, v in self.__dict__.items():
            v.update(*args, **kwargs)

    def get_metrics(self, *args, **kwargs):
        metrics = {}
        for k, v in self.__dict__.items():
            metrics[k] = v.get_metric(*args, **kwargs)
        return metrics

import torch


class MetricsWrapper:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        for k, v in self.__dict__.items():
            v.reset()

    def update(self, *args, **kwargs):
        for k, v in self.__dict__.items():
            v.update(*args, **kwargs)

    def get_metrics(self, *args, **kwargs):
        metrics = dict()
        for k, v in self.__dict__.items():
            metrics[k] = v.get_metric(*args, **kwargs)
        return metrics

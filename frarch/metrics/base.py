import torch

AGGREGATION_MODES = ['mean', 'max', 'min']

class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = list()

    def update(self):
        raise NotImplementedError

    def get_metric(self, mode='mean'):
        if mode not in AGGREGATION_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes: {AGGREGATION_MODES}")
        if mode == 'mean':
            return sum(self.metrics) / len(self.metrics)
        elif mode == 'max':
            return max(self.metrics)
        elif mode == 'min':
            return min(self.metrics)

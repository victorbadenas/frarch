import torch

from .base import Metric


class ClassificationError(Metric):
    def update(self, predictions, truth):
        if predictions.shape[0] != truth.shape[0]:
            raise ValueError(f"mismatched shapes {predictions.shape} != {truth.shape}")
        predictions = torch.argmax(predictions, dim=-1)
        self.metrics.append((predictions != truth).float().mean().item())

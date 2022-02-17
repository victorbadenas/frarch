import torch

from .base import Metric


class ClassificationError(Metric):
    """Classification error metric.

    Example:
        Sample code for use of the ClassificationError metric class:
            model = MyModel()
            error = ClassificationError()
            for batch, labels in dataset:
                predictions = model(batch)
                error.update(predictions, labels)
            print(error.get_metric(mode="mean"))
    """

    def _update(self, predictions: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if predictions.shape[0] != truth.shape[0]:
            raise ValueError(f"mismatched shapes {predictions.shape} != {truth.shape}")
        predictions = torch.argmax(predictions, dim=-1)
        return (predictions != truth).float().mean().item()

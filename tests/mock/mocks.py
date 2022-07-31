from typing import Any
from typing import Tuple

import torch
from torch.utils.data import Dataset

from frarch.modules.metrics.base import Metric


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, inputs):
        return self.fc(inputs)


class MockClassificationDataset(Dataset):
    def __init__(self, n_classes: int = 2, image_shape: Tuple[int] = (10,)):
        super().__init__()
        self.n_classes = n_classes
        self.items = [torch.full(image_shape, i, dtype=torch.float) for i in range(10)]

    def __getitem__(self, index):
        return self.items[index], 0

    def __len__(self):
        return len(self.items)


class MockMetric(Metric):
    def _update(self, data: Any, truth: Any) -> Any:
        return data

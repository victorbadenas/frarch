import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """Small CNN network for FashionMNIST dataset.

    Args:
        out_size (int): Size of the output embedding for the feature extraction
            network. Defaults to 128.
    """

    def __init__(self, out_size: int = 128) -> None:
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=512, out_features=out_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call.

        forward computation for FashionCNN.

        Args:
            x (torch.Tensor): input to the model.

        Returns:
            torch.Tensor: output of the model.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)

        return out


class FashionClassifier(nn.Module):
    """Classifier network for FashionCNN.

    Args:
        embedding_size (int): embedding size from FashionCNN network. Defaults to 128.
        classes (int): number of output classes for the classifier. Defaults to 10.
    """

    def __init__(self, embedding_size: int = 128, classes: int = 10) -> None:
        super(FashionClassifier, self).__init__()
        self.fc = nn.Linear(in_features=embedding_size, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call.

        forward computation for FashionCNN.

        Args:
            x (torch.Tensor): input to the model.

        Returns:
            torch.Tensor: output of the model.
        """
        return self.fc(x)

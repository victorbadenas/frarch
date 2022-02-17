import torch
import torch.nn as nn
import torch.nn.functional as F


class MitCNN(nn.Module):
    """Small CNN network for Mit67 dataset.

    Args:
        input_channels (int): Number of input channels in the input tensor.
            Defaults to 1.
        embedding_size (int): Size of the output embedding for the feature extraction
            network. Defaults to 256.
    """

    def __init__(self, input_channels: int = 1, embedding_size: int = 256) -> None:
        super(MitCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.fc1 = nn.Linear(256, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call.

        forward computation for MitCNN.

        Args:
            x (torch.Tensor): input to the model.

        Returns:
            torch.Tensor: output of the model.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x


class MitCNNClassifier(nn.Module):
    """Classifier network for MitCNN.

    Args:
        embedding_size (int): embedding size from MitCNN network. Defaults to 256.
        classes (int): number of output classes for the classifier. Defaults to 10.
    """

    def __init__(self, embedding_size: int = 256, num_classes: int = 10) -> None:
        super(MitCNNClassifier, self).__init__()
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call.

        forward computation for MitCNNClassifier.

        Args:
            x (torch.Tensor): input to the model.

        Returns:
            torch.Tensor: output of the model.
        """
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

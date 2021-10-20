import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self, input_channels=1, embedding_size=256):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=10, padding="same")
        self.conv2 = nn.Conv2d(32, 32, kernel_size=10, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=10, padding="same")
        self.fc1 = nn.Linear(64 * 7 * 7, embedding_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x


class MNISTClassifier(nn.Module):
    def __init__(self, embedding_size=256, num_classes=10):
        super(MNISTClassifier, self).__init__()
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

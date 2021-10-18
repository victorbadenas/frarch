import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self, out_size=128):
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

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)

        return out


class FashionClassifier(nn.Module):
    def __init__(self, embedding_size=128, classes=10):
        super(FashionClassifier, self).__init__()
        self.fc = nn.Linear(in_features=embedding_size, out_features=classes)

    def forward(self, x):
        return self.fc(x)

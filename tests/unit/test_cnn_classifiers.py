import unittest
from typing import Tuple

import torch

from frarch.models.classification.cnn import FashionClassifier
from frarch.models.classification.cnn import FashionCNN
from frarch.models.classification.cnn import MitCNN
from frarch.models.classification.cnn import MitCNNClassifier
from frarch.models.classification.cnn import MNISTClassifier
from frarch.models.classification.cnn import MNISTCNN
from frarch.models.classification.cnn import vgg11
from frarch.models.classification.cnn import vgg11_bn
from frarch.models.classification.cnn import vgg13
from frarch.models.classification.cnn import vgg13_bn
from frarch.models.classification.cnn import vgg16
from frarch.models.classification.cnn import vgg16_bn
from frarch.models.classification.cnn import vgg19
from frarch.models.classification.cnn import vgg19_bn
from frarch.models.classification.cnn import vggclassifier

VGG_CONFIGS = {
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
}


def forward_model(model: torch.nn.Module, shape: Tuple):
    return model(torch.rand(shape))


class TestCNNClassifiers(unittest.TestCase):
    def test_MNISTCNN_init(self):
        MNISTCNN(input_channels=1, embedding_size=256)

    def test_MNISTCNN_forward(self):
        model = MNISTCNN(input_channels=1, embedding_size=256)
        out = forward_model(model, (2, 1, 28, 28))
        self.assertEqual(out.shape, (2, 256))

    def test_MNISTClassifier_init(self):
        MNISTClassifier(embedding_size=256, num_classes=10)

    def test_MNISTClassifier_forward(self):
        model = MNISTClassifier(embedding_size=256, num_classes=10)
        out = forward_model(model, (2, 256))
        self.assertEqual(out.shape, (2, 10))

    def test_FashionCNN_init(self):
        FashionCNN(out_size=256)

    def test_FashionCNN_forward(self):
        model = FashionCNN(out_size=256)
        out = forward_model(model, (2, 1, 32, 32))
        self.assertEqual(out.shape, (2, 256))

    def test_FashionClassifier_init(self):
        FashionClassifier(embedding_size=256, classes=10)

    def test_FashionClassifier_forward(self):
        model = FashionClassifier(embedding_size=256, classes=10)
        out = forward_model(model, (2, 256))
        self.assertEqual(out.shape, (2, 10))

    def test_MitCNN_init(self):
        MitCNN(input_channels=1, embedding_size=256)

    def test_MitCNN_forward(self):
        model = MitCNN(input_channels=1, embedding_size=256)
        out = forward_model(model, (2, 1, 32, 32))
        self.assertEqual(out.shape, (2, 256))

    def test_MitCNNClassifier_init(self):
        MitCNNClassifier(embedding_size=256, num_classes=10)

    def test_MitCNNClassifier_forward(self):
        model = MitCNNClassifier(embedding_size=256, num_classes=10)
        out = forward_model(model, (2, 256))
        self.assertEqual(out.shape, (2, 10))

    def test_VGGClassifier_init(self):
        vggclassifier(False, num_classes=100)

    def test_VGGClassifier_forward(self):
        model = vggclassifier(False, 100)
        out = forward_model(model, (2, 25088))
        self.assertEqual(out.shape, (2, 100))

    def test_vggconfigurations_init(self):
        for conf_name, conf_fn in VGG_CONFIGS.items():
            try:
                conf_fn()
            except Exception as e:
                raise type(e)(f"Exception thrown for {conf_name}") from e

    def test_vggconfigurations_forward(self):
        for conf_name, conf_fn in VGG_CONFIGS.items():
            model = conf_fn()
            out = forward_model(model, (2, 3, 244, 244))

            self.assertEqual(
                out.shape, (2, 25088), msg=f"shape mismatch for {conf_name} arch"
            )


if __name__ == "__main__":
    unittest.main()

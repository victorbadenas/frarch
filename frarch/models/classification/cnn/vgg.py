"""
VGG definition. Slightly modified pytorch implementation.

:Description: VGG

:Authors: victor badenas (victor.badenas@gmail.com),
          pytorch.org

:Version: 0.1.0
:Created on: 21/07/2021 19:00
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Union, cast

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "VGG",
]

l11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
l13 = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
l16 = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]
l19 = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]

CONFIGURATIONS: Dict[str, List[Union[str, int]]] = {
    "11": l11,
    "13": l13,
    "16": l16,
    "19": l19,
}

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(
        self,
        layers_cfg: List[Union[str, int]],
        batch_norm: bool = True,
        init_weights: bool = True,
        pretrained: bool = False,
        padding: str = "same",
    ) -> None:
        super(VGG, self).__init__()
        self.layers_cfg = layers_cfg
        self.batch_norm = batch_norm
        self.padding = padding
        self.features = self.make_layers(layers_cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7) if padding == "same" else (3, 3))
        if init_weights:
            self._initialize_weights()
        elif pretrained:
            self._load_pretrained(layers_cfg, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(
        self, layers_cfg: List[Union[str, int]], batch_norm: bool = False
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        for v in layers_cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=self.padding)
                layers.append(conv2d)
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def _load_pretrained(self, layers_cfg, batch_norm):
        arch = "vgg"
        for model_depth, layers_preconfig in CONFIGURATIONS.items():
            if layers_cfg == layers_preconfig:
                arch += model_depth
                if batch_norm:
                    arch += "_bn"
                state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
                features_state_dict, _ = split_state_dict(
                    state_dict, "features", "classifier"
                )
                self.load_state_dict(features_state_dict)
                return
        else:
            logging.warning(
                "no match found for pretrained models in the pytorch repository"
            )
            logging.warning("possible values:")
            for k, v in CONFIGURATIONS.items():
                logging.warning(f"vgg{k}: {v}")
            logging.warning("pretrained model not loaded. Starting from scratch.")


class VGGClassifier(nn.Module):
    def __init__(
        self, num_classes, init_weights=True, pretrained=False, arch="", padding="same"
    ):
        super(VGGClassifier, self).__init__()
        self.in_features = 25088 if padding == "same" else 4608
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        elif pretrained and arch != "":
            self._load_pretrained(arch)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self, arch):
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        _, classifier_state_dict = split_state_dict(
            state_dict, "features", "classifier"
        )
        self.load_state_dict(classifier_state_dict)

    def forward(self, x):
        x = self.classifier(x)
        return x


def split_state_dict(state_dict, *search_strings):
    results = [OrderedDict() for _ in range(len(search_strings))]
    for i, string in enumerate(search_strings):
        for k in state_dict.keys():
            if string in k:
                results[i][k] = state_dict.get(k)
    return results


def _vgg(cfg: str, batch_norm: bool, pretrained: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(
        CONFIGURATIONS[cfg], batch_norm=batch_norm, pretrained=pretrained, **kwargs
    )
    return model


def vggclassifier(
    pretrained: bool, num_classes: int = 1000, arch: str = "", **kwargs: Any
) -> VGGClassifier:
    if pretrained and arch not in model_urls:
        logging.warning(
            "No architecture specified or arch does not exist, not loading model"
        )
        logging.warning(f"Pretrained model possibilities: {list(model_urls.keys())}")
        pretrained = False

    if pretrained and num_classes != 1000:
        logging.warning(
            "when loading pretrained classifiers, num_classes must"
            " be 1000, defaulting to random weights"
        )
        pretrained = False

    if pretrained:
        kwargs["init_weights"] = False

    model = VGGClassifier(num_classes, pretrained=pretrained, arch=arch, **kwargs)
    return model


def vgg11(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model from <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("11", False, pretrained, **kwargs)


def vgg11_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model with batch normalization <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("11", True, pretrained, **kwargs)


def vgg13(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("13", False, pretrained, **kwargs)


def vgg13_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model with batch normalization <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("13", True, pretrained, **kwargs)


def vgg16(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("16", False, pretrained, **kwargs)


def vgg16_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model with batch normalization <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("16", True, pretrained, **kwargs)


def vgg19(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("19", False, pretrained, **kwargs)


def vgg19_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer modelith batch normalization <https://arxiv.org/pdf/1409.1556.pdf>`_.

    The required minimum input size of the model is 32x32.

    Args
    ----
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg("19", True, pretrained, **kwargs)

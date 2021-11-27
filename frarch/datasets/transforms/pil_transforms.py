import random
from typing import Callable, Iterable, Union

import PIL.Image as im
import PIL.ImageEnhance as ie
import torch
from PIL.Image import Image

NoneType = type(None)


class RandomFlip(object):
    """Randomly flips the given PIL.Image.

    Probability of 0.25 horizontal, 0.25 vertical, 0.5 as is
    """

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        dispatcher = {
            0: img,
            1: img,
            2: img.transpose(im.FLIP_LEFT_RIGHT),
            3: img.transpose(im.FLIP_TOP_BOTTOM),
        }

        return dispatcher[random.randint(0, 3)]


class RandomRotate(object):
    """Randomly rotate the given PIL.Image.

    Probability of 1/6 90°, 1/6 180°, 1/6 270°, 1/2 as is.
    """

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        dispatcher = {
            0: img,
            1: img,
            2: img,
            3: img.transpose(im.ROTATE_90),
            4: img.transpose(im.ROTATE_180),
            5: img.transpose(im.ROTATE_270),
        }

        return dispatcher[random.randint(0, 5)]


class PILColorBalance(object):
    def __init__(self, var: float):
        if not isinstance(var, float):
            raise ValueError(f"{self.__class__.__name__}.var must be a float value")
        if var < 0 or var > 1:
            raise ValueError(
                f"{self.__class__.__name__}.var must be a float value between 0 and 1"
            )
        self.var = var

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)


class PILContrast(object):
    def __init__(self, var: float):
        if not isinstance(var, float):
            raise ValueError(f"{self.__class__.__name__}.var must be a float value")
        if var < 0 or var > 1:
            raise ValueError(
                f"{self.__class__.__name__}.var must be a float value between 0 and 1"
            )
        self.var = var

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):
    def __init__(self, var: float):
        if not isinstance(var, float):
            raise ValueError(f"{self.__class__.__name__}.var must be a float value")
        if var < 0 or var > 1:
            raise ValueError(
                f"{self.__class__.__name__}.var must be a float value between 0 and 1"
            )
        self.var = var

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)


class PILSharpness(object):
    def __init__(self, var: float):
        if not isinstance(var, float):
            raise ValueError(f"{self.__class__.__name__}.var must be a float value")
        if var < 0 or var > 1:
            raise ValueError(
                f"{self.__class__.__name__}.var must be a float value between 0 and 1"
            )
        self.var = var

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)


class RandomOrder(object):
    """Composes several transforms together in random order."""

    def __init__(self, transforms: Union[Iterable[Callable], NoneType]):
        if not isinstance(transforms, (Iterable, NoneType)):
            raise ValueError("transforms must be an iterable object")
        if transforms is not None:
            if len(transforms) == 0:
                transforms = None
            elif not all(isinstance(x, Callable) for x in transforms):
                raise ValueError("all objects in transforms must be callable")
        self.transforms = transforms

    def __call__(self, img: Image):
        if not isinstance(img, Image):
            raise ValueError(f"img is {type(img)} not a PIL.Image object")

        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class PowerPIL(RandomOrder):
    def __init__(
        self,
        rotate: bool = True,
        flip: bool = True,
        colorbalance: float = 0.4,
        contrast: float = 0.4,
        brightness: float = 0.4,
        sharpness: float = 0.4,
    ):
        self._check_parameters(
            rotate, flip, colorbalance, contrast, brightness, sharpness
        )
        self.transforms = []
        if rotate:
            self.transforms.append(RandomRotate())
        if flip:
            self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))

    @staticmethod
    def _check_parameters(rotate, flip, colorbalance, contrast, brightness, sharpness):
        if not isinstance(rotate, bool):
            raise ValueError("rotate must be boolean")
        if not isinstance(flip, bool):
            raise ValueError("flip must be boolean")

        if isinstance(colorbalance, float):
            if not 0 <= colorbalance <= 1:
                raise ValueError("colorbalance must be float between 0 and 1")
        else:
            raise ValueError("colorbalance must be float between 0 and 1")

        if isinstance(contrast, float):
            if not 0 <= contrast <= 1:
                raise ValueError("contrast must be float between 0 and 1")
        else:
            raise ValueError("contrast must be float between 0 and 1")

        if isinstance(brightness, float):
            if not 0 <= brightness <= 1:
                raise ValueError("brightness must be float between 0 and 1")
        else:
            raise ValueError("brightness must be float between 0 and 1")

        if isinstance(sharpness, float):
            if not 0 <= sharpness <= 1:
                raise ValueError("sharpness must be float between 0 and 1")
        else:
            raise ValueError("sharpness must be float between 0 and 1")

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import frarch.datasets.transforms as t

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"


class TestImageTransforms(unittest.TestCase):

    SYM_IMG = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))

    ASYM_IMG = Image.fromarray(np.random.randint(0, 255, (50, 100, 3)).astype(np.uint8))

    def test_RandomFlip_symetric(self):
        for _ in range(10):
            out_image = t.RandomFlip()(self.SYM_IMG)
            self.assertEqual(out_image.size, self.SYM_IMG.size)

    def test_RandomFlip_asymetric(self):
        for _ in range(10):
            out_image = t.RandomFlip()(self.ASYM_IMG)
            self.assertEqual(out_image.size, self.ASYM_IMG.size)

    def test_RandomFlip_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.RandomFlip()(np.random.randint((10, 10, 3)))

    def test_RandomRotate_symetric(self):
        for _ in range(10):
            out_image = t.RandomRotate()(self.SYM_IMG)
            self.assertEqual(out_image.size, self.SYM_IMG.size)

    def test_RandomRotate_asymetric(self):
        possible_sizes = [self.ASYM_IMG.size, tuple(reversed(self.ASYM_IMG.size))]
        for _ in range(10):
            out_image = t.RandomRotate()(self.ASYM_IMG)
            self.assertIn(out_image.size, possible_sizes)

    def test_RandomRotate_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.RandomRotate()(np.random.randint((10, 10, 3)))

    def test_PILColorBalance(self):
        out_image = t.PILColorBalance(0.1)(self.SYM_IMG)
        self.assertEqual(out_image.size, self.SYM_IMG.size)

    def test_PILColorBalance_negative_alpha(self):
        with self.assertRaises(ValueError):
            t.PILColorBalance(-1.0)

    def test_PILColorBalance_too_big_alpha(self):
        with self.assertRaises(ValueError):
            t.PILColorBalance(3.0)

    def test_PILColorBalance_not_float_alpha(self):
        with self.assertRaises(ValueError):
            t.PILColorBalance(int(1))

    def test_PILColorBalance_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.PILColorBalance(0.1)(np.random.randint((10, 10, 3)))

    def test_PILContrast(self):
        out_image = t.PILContrast(0.1)(self.SYM_IMG)
        self.assertEqual(out_image.size, self.SYM_IMG.size)

    def test_PILContrast_negative_alpha(self):
        with self.assertRaises(ValueError):
            t.PILContrast(-1.0)

    def test_PILContrast_too_big_alpha(self):
        with self.assertRaises(ValueError):
            t.PILContrast(3.0)

    def test_PILContrast_not_float_alpha(self):
        with self.assertRaises(ValueError):
            t.PILContrast(int(1))

    def test_PILContrast_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.PILContrast(0.1)(np.random.randint((10, 10, 3)))

    def test_PILBrightness(self):
        out_image = t.PILBrightness(0.1)(self.SYM_IMG)
        self.assertEqual(out_image.size, self.SYM_IMG.size)

    def test_PILBrightness_negative_alpha(self):
        with self.assertRaises(ValueError):
            t.PILBrightness(-1.0)

    def test_PILBrightness_too_big_alpha(self):
        with self.assertRaises(ValueError):
            t.PILBrightness(3.0)

    def test_PILBrightness_not_float_alpha(self):
        with self.assertRaises(ValueError):
            t.PILBrightness(int(1))

    def test_PILBrightness_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.PILBrightness(0.1)(np.random.randint((10, 10, 3)))

    def test_PILSharpness(self):
        out_image = t.PILSharpness(0.1)(self.SYM_IMG)
        self.assertEqual(out_image.size, self.SYM_IMG.size)

    def test_PILSharpness_negative_alpha(self):
        with self.assertRaises(ValueError):
            t.PILSharpness(-1.0)

    def test_PILSharpness_too_big_alpha(self):
        with self.assertRaises(ValueError):
            t.PILSharpness(3.0)

    def test_PILSharpness_not_float_alpha(self):
        with self.assertRaises(ValueError):
            t.PILSharpness(int(1))

    def test_PILSharpness_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.PILSharpness(0.1)(np.random.randint((10, 10, 3)))

    def test_RandomOrder(self):
        for i in range(2):
            called_fn = []

            def fn0(*args):
                called_fn.append(0)
                return args

            def fn1(*args):
                called_fn.append(1)
                return args

            torch.random.manual_seed(i)
            ro = t.RandomOrder([fn0, fn1])
            ro(self.SYM_IMG)
            if i == 0:
                self.assertEqual(called_fn, [0, 1])
            else:
                self.assertEqual(called_fn, [1, 0])

    def test_RandomOrder_None(self):
        ro = t.RandomOrder(None)
        self.assertEqual(ro(self.SYM_IMG), self.SYM_IMG)

    def test_RandomOrder_empty_list(self):
        ro = t.RandomOrder([])
        self.assertEqual(ro(self.SYM_IMG), self.SYM_IMG)

    def test_RandomOrder_transforms_not_iterable(self):
        def fn(*args):
            return args

        with self.assertRaises(ValueError):
            t.RandomOrder(fn)

    def test_RandomOrder_transforms_not_callable(self):
        with self.assertRaises(ValueError):
            t.RandomOrder([0, 1])

    def test_RandomOrder_not_PIL_image(self):
        with self.assertRaises(ValueError):
            t.RandomOrder(None)(np.random.randint((10, 10, 3)))

    def test_PowerPil_rotate_not_bool(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(rotate="True")

    def test_PowerPil_flip_not_bool(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(flip="True")

    def test_PowerPil_colorbalance_not_float(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(colorbalance="0.5")

    def test_PowerPil_colorbalance_not_in_range(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(colorbalance=1.1)

    def test_PowerPil_contrast_not_float(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(contrast="0.5")

    def test_PowerPil_contrast_not_in_range(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(contrast=1.1)

    def test_PowerPil_brightness_not_float(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(brightness="0.5")

    def test_PowerPil_brightness_not_in_range(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(brightness=1.1)

    def test_PowerPil_sharpness_not_float(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(sharpness="0.5")

    def test_PowerPil_sharpness_not_in_range(self):
        with self.assertRaises(ValueError):
            t.PowerPIL(sharpness=1.1)

    def test_PowerPIL(self):
        pp = t.PowerPIL(
            rotate=True,
            flip=True,
            colorbalance=0.4,
            contrast=0.4,
            brightness=0.4,
            sharpness=0.4,
        )
        self.assertEqual(len(pp.transforms), 6)


if __name__ == "__main__":
    unittest.main()

import unittest


class DummyTest(unittest.TestCase):
    def test_dummy(self):
        a = 1
        self.assertTrue(1, a)


if __name__ == "__main__":
    unittest.main()

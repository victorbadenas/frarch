import unittest

import torch

from frarch.modules import metrics
from frarch.modules.metrics.base import Metric


class MockMetric(Metric):
    def update(self, data):
        self.metrics.append(data)


class TestMetrics(unittest.TestCase):
    def test_metricBase_constructor(self):
        m = Metric()
        self.assertEqual(len(m.metrics), 0)

    def test_metricBase_length(self):
        m = Metric()
        self.assertEqual(len(m.metrics), len(m))

    def test_metricBase_virtual_update(self):
        with self.assertRaises(NotImplementedError):
            Metric().update()

    def test_metricBase_get_metric_mean(self):
        m = Metric()
        m.metrics = [0, 1, 2]
        self.assertEqual(m.get_metric(mode="mean"), 1)

    def test_metricBase_get_metric_max(self):
        m = Metric()
        m.metrics = [0, 1, 2]
        self.assertEqual(m.get_metric(mode="max"), 2)

    def test_metricBase_get_metric_min(self):
        m = Metric()
        m.metrics = [0, 1, 2]
        self.assertEqual(m.get_metric(mode="min"), 0)

    def test_metricBase_agg_mode_not_valid(self):
        m = Metric()
        m.metrics = [0, 1, 2]
        with self.assertRaises(ValueError):
            m.get_metric(mode="not-valid")

    def test_metricBase_empty_metric(self):
        m = Metric()
        self.assertEquals(m.get_metric(), 0.0)

    def test_classification_error_update_accurate(self):
        predictions = torch.Tensor([[0, 1], [1, 0]])
        truth = torch.Tensor([0, 1])
        m = metrics.ClassificationError()
        m.update(predictions, truth)
        self.assertEquals(m.get_metric(), 1.0)

    def test_classification_error_update_inaccurate(self):
        predictions = torch.Tensor([[0, 1], [1, 0]])
        truth = torch.Tensor([1, 0])
        m = metrics.ClassificationError()
        m.update(predictions, truth)
        self.assertEquals(m.get_metric(), 0.0)

    def test_classification_error_mismatch(self):
        predictions = torch.Tensor([[0, 1], [1, 0]])
        truth = torch.Tensor([1, 0, 0])
        m = metrics.ClassificationError()
        with self.assertRaises(ValueError):
            m.update(predictions, truth)

    def test_metricsWrapper_init(self):
        mw = metrics.MetricsWrapper(metric0=MockMetric(), metric1=MockMetric())
        self.assertTrue(hasattr(mw, "metric0"))
        self.assertTrue(hasattr(mw, "metric1"))

    def test_metricsWrapper_not_metric(self):
        with self.assertRaises(ValueError):
            metrics.MetricsWrapper(metric0=MockMetric(), metric1="not-a-metric")

    def test_metricsWrapper_reset(self):
        mw = metrics.MetricsWrapper(metric0=MockMetric(), metric1=MockMetric())
        mw.metric0.metrics = [0, 1, 2]
        mw.metric1.metrics = [0]
        mw.reset()
        self.assertEqual(len(mw.metric0), 0)
        self.assertEqual(len(mw.metric1), 0)

    def test_metricsWrapper_update(self):
        mw = metrics.MetricsWrapper(metric0=MockMetric(), metric1=MockMetric())
        mw.update(1)
        self.assertEquals(mw.metric0.metrics, [1])
        self.assertEquals(mw.metric1.metrics, [1])

    def test_metricsWrapper_get_metrics_mean(self):
        mw = metrics.MetricsWrapper(metric0=MockMetric(), metric1=MockMetric())
        mw.update(0)
        mw.update(1)
        m = mw.get_metrics(mode="mean")
        self.assertDictEqual(m, {"metric0": 0.5, "metric1": 0.5})

    def test_metricsWrapper_get_metrics_max(self):
        mw = metrics.MetricsWrapper(metric0=MockMetric(), metric1=MockMetric())
        mw.update(0)
        mw.update(1)
        m = mw.get_metrics(mode="max")
        self.assertDictEqual(m, {"metric0": 1, "metric1": 1})

    def test_metricsWrapper_get_metrics_min(self):
        mw = metrics.MetricsWrapper(metric0=MockMetric(), metric1=MockMetric())
        mw.update(0)
        mw.update(1)
        m = mw.get_metrics(mode="min")
        self.assertDictEqual(m, {"metric0": 0, "metric1": 0})


if __name__ == "__main__":
    unittest.main()

import numpy as np

from devp_ohio.metrics import MetricBundle, pbias


def test_metric_bundle_returns_expected_keys() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 2.8, 4.2])
    metrics = MetricBundle.compute(y_true, y_pred)
    assert set(metrics) == {"RMSE", "NSE", "PBIAS", "KGE"}
    assert metrics["RMSE"] >= 0.0


def test_metric_bundle_clamps_negative_predictions() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([-5.0, 2.0, 3.0])
    metrics = MetricBundle.compute(y_true, y_pred)
    assert np.isfinite(metrics["NSE"])
    assert pbias(y_true, y_pred) > 0.0

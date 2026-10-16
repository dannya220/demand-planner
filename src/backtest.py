"""Метрики прогнозов на одинаковых строках."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def regression_metrics(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.ndim != 1 or actual.shape != prediction.shape or actual.size == 0:
        raise ValueError("Actual values and predictions must be nonempty matching vectors.")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Actual values and predictions must be finite on every evaluation row.")
    if (actual < 0).any():
        raise ValueError("Sales cannot be negative.")
    errors = prediction - actual
    volume = actual.sum()
    return {
        "MAE": float(np.abs(errors).mean()),
        "WAPE": float(np.abs(errors).sum() / volume) if volume > 0 else float("nan"),
        "Bias": float(errors.sum() / volume) if volume > 0 else float("nan"),
    }


def evaluate_forecasts(
    data: pd.DataFrame,
    methods: Mapping[str, str],
    target: str = "weekly_sales",
) -> pd.DataFrame:
    """Evaluate every method on all supplied rows, without silently dropping missing predictions."""
    if not methods:
        raise ValueError("Provide at least one forecast method.")
    rows = [
        {"Метод": name, **regression_metrics(data[target], data[column])}
        for name, column in methods.items()
    ]
    return pd.DataFrame(rows).set_index("Метод")

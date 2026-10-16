"""Недельные признаки из прошлых продаж."""

import numpy as np
import pandas as pd


SERIES_KEYS = ["store_id", "item_id"]
CATEGORICAL_FEATURES = ["store_id", "item_id"]
NUMERIC_FEATURES = [
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "mean_4w", "mean_8w", "std_4w", "month", "week_of_year",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
BASELINE_METHODS = {
    "Предыдущая неделя": "prediction",
    "Среднее за 4 недели": "prediction_mean_4w",
    "Медиана за 4 недели": "prediction_median_4w",
}


def add_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add the pilot's lag features and baselines without using current-week sales."""
    if weekly.empty or weekly.duplicated([*SERIES_KEYS, "week_end"]).any():
        raise ValueError("Weekly rows must be nonempty and unique per series and date.")
    data = weekly.sort_values([*SERIES_KEYS, "week_end"]).reset_index(drop=True).copy()
    if data["week_end"].isna().any() or data["weekly_sales"].isna().any():
        raise ValueError("Dates and sales must be observed.")
    groups = data.groupby(SERIES_KEYS, sort=False, observed=True)
    if not groups["week_end"].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("Weekly series have gaps; restore calendar weeks before making lags.")
    sales = groups["weekly_sales"]
    for lag in (1, 2, 3, 4):
        data[f"lag_{lag}w"] = sales.shift(lag)
    data["mean_4w"] = sales.transform(lambda s: s.shift(1).rolling(4).mean())
    data["mean_8w"] = sales.transform(lambda s: s.shift(1).rolling(8).mean())
    data["std_4w"] = sales.transform(lambda s: s.shift(1).rolling(4).std())
    data["month"] = data["week_end"].dt.month.astype("int8")
    data["week_of_year"] = data["week_end"].dt.isocalendar().week.astype("int8")
    data["prediction"] = data["lag_1w"]
    data["prediction_mean_4w"] = data["mean_4w"]
    data["prediction_median_4w"] = sales.transform(lambda s: s.shift(1).rolling(4).median())
    data["forecast_origin"] = data["week_end"] - pd.Timedelta(days=7)
    data["known_at_origin"] = data["first_price_date"].le(data["forecast_origin"])
    return data


def split_by_time(
    data: pd.DataFrame,
    validation_weeks: int = 8,
    reference_weeks: int = 4,
) -> pd.DataFrame:
    """Keep the pilot dates comparable; the final reference period has been viewed."""
    if validation_weeks < 1 or reference_weeks < 1:
        raise ValueError("Validation and reference periods must each contain at least one week.")
    week_dates = data["week_end"].drop_duplicates().sort_values().tolist()
    if len(week_dates) <= validation_weeks + reference_weeks:
        raise ValueError("Not enough history remains for training.")
    train_end = week_dates[-validation_weeks - reference_weeks - 1]
    validation_end = week_dates[-reference_weeks - 1]
    result = data.copy()
    result["split"] = pd.Categorical(
        np.select(
            [result["week_end"].le(train_end), result["week_end"].le(validation_end)],
            ["train", "validation"],
            default="reference",
        ),
        categories=["train", "validation", "reference"],
        ordered=True,
    )
    return result

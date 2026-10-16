import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.backtest import evaluate_forecasts, regression_metrics
from src.data import build_panel, build_weekly_table
from src.features import BASELINE_METHODS, FEATURE_COLUMNS, add_weekly_features, split_by_time


class WeeklyAggregationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2011-01-29", periods=30)
        self.calendar = pd.DataFrame({
            "d": [f"d_{i + 1}" for i in range(30)],
            "date": dates,
            "wm_yr_wk": 11101 + np.arange(30) // 7,
        })
        rows = []
        for item, department in [("A", "D1"), ("B", "D2")]:
            row = dict(id=item + "_CA_1_validation", item_id=item, dept_id=department,
                       cat_id="FOODS", store_id="CA_1", state_id="CA")
            row.update({f"d_{i + 1}": (0 if item == "A" and i < 7 else i + 1) for i in range(23)})
            rows.append(row)
        self.sales = pd.DataFrame(rows)
        self.prices = pd.DataFrame([
            {"store_id": "CA_1", "item_id": item, "wm_yr_wk": week,
             "sell_price": np.nan if item == "A" and week == 11101 else 2.0}
            for item in ["A", "B"] for week in range(11101, 11106)
        ])

    def test_direct_weekly_matches_daily_panel_with_partial_weeks(self):
        panel = build_panel(self.calendar, self.sales, self.prices)
        expected = (
            panel.groupby([pd.Grouper(key="date", freq="W-SUN"), "store_id", "item_id"])
            .agg(weekly_sales=("sales", "sum"), days_in_week=("date", "nunique"))
            .reset_index().rename(columns={"date": "week_end"})
        )
        expected = expected.loc[expected.days_in_week.eq(7)].sort_values(
            ["store_id", "item_id", "week_end"]
        ).reset_index(drop=True)
        result = build_weekly_table(self.calendar, self.sales, self.prices)
        for column in ["store_id", "item_id"]:
            result[column] = result[column].astype(object)
        expected = expected[["store_id", "item_id", "week_end", "weekly_sales"]]
        actual = result[expected.columns]
        assert_frame_equal(actual, expected, check_dtype=False)
        self.assertEqual(len(result), 6)
        first_price = result.groupby("item_id", observed=True).first_price_date.first()
        self.assertEqual(first_price["A"], pd.Timestamp("2011-02-05"))
        self.assertEqual(first_price["B"], pd.Timestamp("2011-01-29"))

    def test_daily_column_order_does_not_change_aggregation(self):
        expected = build_weekly_table(self.calendar, self.sales, self.prices)
        daily = [c for c in self.sales.columns if c.startswith("d_")]
        identifiers = [c for c in self.sales.columns if not c.startswith("d_")]
        shuffled = self.sales[identifiers + list(reversed(daily))]
        actual = build_weekly_table(self.calendar, shuffled, self.prices)
        assert_frame_equal(actual, expected)

    def test_duplicate_price_keys_are_rejected(self):
        duplicated = pd.concat([self.prices, self.prices.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_weekly_table(self.calendar, self.sales, duplicated)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.weekly = pd.DataFrame([
            dict(store_id="CA_1", item_id=item, week_end=date,
                 weekly_sales=number + (20 if item == "B" else 0),
                 first_price_date=pd.Timestamp("2020-01-01"))
            for item in ["A", "B"]
            for number, date in enumerate(pd.date_range("2020-01-05", periods=16, freq="W-SUN"))
        ])

    def test_current_and_future_sales_do_not_change_earlier_features(self):
        cutoff = pd.Timestamp("2020-03-08")
        original = add_weekly_features(self.weekly)
        changed = self.weekly.copy()
        changed.loc[changed.week_end.ge(cutoff), "weekly_sales"] += 10000
        updated = add_weekly_features(changed)
        columns = FEATURE_COLUMNS + list(BASELINE_METHODS.values())
        assert_frame_equal(
            original.loc[original.week_end.le(cutoff), columns],
            updated.loc[updated.week_end.le(cutoff), columns],
        )
        after_cutoff = original.week_end.eq(cutoff + pd.Timedelta(days=7))
        self.assertTrue((updated.loc[after_cutoff, "lag_1w"] > original.loc[after_cutoff, "lag_1w"]).all())

    def test_one_items_history_does_not_change_other_items_features(self):
        original = add_weekly_features(self.weekly)
        changed = self.weekly.copy()
        changed.loc[changed.item_id.eq("A"), "weekly_sales"] += 500
        updated = add_weekly_features(changed)
        columns = FEATURE_COLUMNS + list(BASELINE_METHODS.values())
        assert_frame_equal(original.loc[original.item_id.eq("B"), columns],
                           updated.loc[updated.item_id.eq("B"), columns])

    def test_missing_calendar_week_is_rejected(self):
        gap = self.weekly.drop(self.weekly.index[4])
        with self.assertRaises(ValueError):
            add_weekly_features(gap)

    def test_time_splits_are_ordered_and_disjoint(self):
        result = split_by_time(self.weekly, validation_weeks=3, reference_weeks=2)
        train = result.loc[result.split.eq("train"), "week_end"]
        valid = result.loc[result.split.eq("validation"), "week_end"]
        reference = result.loc[result.split.eq("reference"), "week_end"]
        self.assertLess(train.max(), valid.min())
        self.assertLess(valid.max(), reference.min())
        self.assertEqual(valid.nunique(), 3)
        self.assertEqual(reference.nunique(), 2)


class MetricTests(unittest.TestCase):
    def test_missing_predictions_cannot_silently_reduce_evaluation_rows(self):
        data = pd.DataFrame({"weekly_sales": [0, 3], "prediction": [1, np.nan]})
        with self.assertRaises(ValueError):
            evaluate_forecasts(data, {"baseline": "prediction"})

    def test_zero_actual_volume_leaves_percentage_metrics_undefined(self):
        metrics = regression_metrics([0, 0], [0, 2])
        self.assertEqual(metrics["MAE"], 1.0)
        self.assertTrue(np.isnan(metrics["WAPE"]))
        self.assertTrue(np.isnan(metrics["Bias"]))


if __name__ == "__main__":
    unittest.main()

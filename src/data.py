"""Load M5 and build daily or weekly product-store tables."""

from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
import shutil
import zipfile

import numpy as np
import pandas as pd


M5_URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"
M5_MD5 = "86f57416a314197f40a17cc6fc60cbb4"
RAW_FILES = ("calendar.csv", "sales_train_validation.csv", "sell_prices.csv")
IDENTIFIERS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_m5(raw_dir: Path) -> pd.DataFrame:
    """Verify the published archive checksum and extract only the three CSVs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "m5-forecasting-accuracy.zip"
    if not archive_path.exists():
        partial = archive_path.with_suffix(".zip.part")
        request = Request(M5_URL, headers={"User-Agent": "demand-planner/0.1"})
        print("Downloading M5 archive...", flush=True)
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if file_md5(partial) != M5_MD5:
            raise ValueError("Archive checksum differs from the published Zenodo checksum.")
        partial.replace(archive_path)
    elif file_md5(archive_path) != M5_MD5:
        raise ValueError("Existing archive has an unexpected checksum; it was not overwritten.")

    metadata = []
    with zipfile.ZipFile(archive_path) as archive:
        for filename in RAW_FILES:
            matches = [info for info in archive.infolist() if Path(info.filename).name == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member for {filename}.")
            member = matches[0]
            destination = raw_dir / filename
            if not destination.exists():
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
            # Check existing files against their original archive bytes too.
            with archive.open(member) as source:
                expected_md5 = hashlib.md5(source.read()).hexdigest()
            if destination.stat().st_size != member.file_size or file_md5(destination) != expected_md5:
                raise ValueError(f"Raw file differs from the original archive: {filename}")
            metadata.append({"file": filename, "bytes": destination.stat().st_size, "md5": expected_md5})

    manifest = {"source": M5_URL, "record": "https://zenodo.org/records/12636070", "archive_md5": M5_MD5, "files": metadata}
    (raw_dir / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pd.DataFrame(metadata)


def load_subset(raw_dir: Path, store_id: str = "CA_1", category: str | None = "FOODS", n_items: int | None = 100, seed: int = 42):
    """Read one store in chunks. None selects all categories or all matching items."""
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    sales_path = raw_dir / "sales_train_validation.csv"
    columns = pd.read_csv(sales_path, nrows=0).columns
    day_columns = [column for column in columns if column.startswith("d_")]
    if not day_columns or not set(IDENTIFIERS).issubset(columns):
        raise ValueError("Unexpected M5 sales schema.")
    dtypes = {column: "int32" for column in day_columns}
    dtypes.update({column: "string" for column in IDENTIFIERS})
    store_parts = []
    sales_rows = 0
    for chunk in pd.read_csv(sales_path, dtype=dtypes, chunksize=5000):
        sales_rows += len(chunk)
        selected = chunk.loc[chunk["store_id"].eq(store_id)]
        if not selected.empty:
            store_parts.append(selected.copy())
    if not store_parts:
        raise ValueError(f"Store not found: {store_id}")
    store_sales = pd.concat(store_parts, ignore_index=True)
    candidates = store_sales if category is None else store_sales.loc[store_sales["cat_id"].eq(category)]
    item_ids = np.array(sorted(candidates["item_id"].unique()), dtype=object)
    if n_items is None:
        if len(item_ids) == 0:
            raise ValueError("No products match the requested selection.")
        selected_ids = item_ids.tolist()
    else:
        if len(item_ids) < n_items or n_items < 1:
            raise ValueError("The requested subset size is unavailable.")
        selected_ids = sorted(np.random.default_rng(seed).choice(item_ids, size=n_items, replace=False).tolist())
    sales = candidates.loc[candidates["item_id"].isin(selected_ids)].copy().reset_index(drop=True)

    price_parts = []
    price_rows = 0
    price_dtypes = {"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float64"}
    for chunk in pd.read_csv(raw_dir / "sell_prices.csv", dtype=price_dtypes, chunksize=200000):
        price_rows += len(chunk)
        selected = chunk.loc[chunk["store_id"].eq(store_id) & chunk["item_id"].isin(selected_ids)]
        if not selected.empty:
            price_parts.append(selected.copy())
    if not price_parts:
        raise ValueError("No prices found for the selected products.")
    prices = pd.concat(price_parts, ignore_index=True)
    source_shapes = pd.DataFrame([
        {"table": "calendar", "rows": len(calendar), "columns": len(calendar.columns)},
        {"table": "sales_train_validation", "rows": sales_rows, "columns": len(columns)},
        {"table": "sell_prices", "rows": price_rows, "columns": 4},
    ])
    return calendar, sales, prices, source_shapes


def build_panel(calendar: pd.DataFrame, sales: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per product, store and date; no filling unknown prices or demand."""
    if sales.empty or sales["id"].duplicated().any():
        raise ValueError("Sales series must be non-empty and uniquely identified.")
    if calendar["d"].duplicated().any() or calendar["date"].duplicated().any():
        raise ValueError("Calendar days must be unique.")
    price_keys = ["store_id", "item_id", "wm_yr_wk"]
    if prices.duplicated(price_keys).any():
        raise ValueError("Prices must be unique for each store-item-week.")
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    panel = sales.melt(id_vars=list(IDENTIFIERS), value_vars=day_columns, var_name="d", value_name="sales")
    panel = panel.merge(calendar, on="d", how="left", validate="many_to_one")
    panel = panel.merge(prices, on=price_keys, how="left", validate="many_to_one")
    if len(panel) != len(sales) * len(day_columns) or panel.duplicated(["item_id", "store_id", "date"]).any():
        raise ValueError("Unexpected panel size or duplicate product-date rows.")
    if panel["date"].isna().any() or panel["sales"].isna().any() or panel["sales"].lt(0).any():
        raise ValueError("Sales and dates must be observed; sales must be nonnegative.")
    if panel["sell_price"].dropna().le(0).any():
        raise ValueError("Observed selling prices must be positive.")
    panel["price_available"] = panel["sell_price"].notna()
    panel["day_index"] = panel["d"].str.removeprefix("d_").astype("int32")
    return panel.sort_values(["item_id", "date"]).reset_index(drop=True)


def build_weekly_table(calendar: pd.DataFrame, sales: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Sum complete Monday–Sunday weeks directly from the wide sales table."""
    series_keys = ["store_id", "item_id"]
    if sales.empty or sales.duplicated(series_keys).any():
        raise ValueError("Sales must contain one nonempty row per store and item.")
    if calendar["d"].duplicated().any() or calendar["date"].duplicated().any():
        raise ValueError("Calendar days must be unique.")
    price_keys = ["store_id", "item_id", "wm_yr_wk"]
    if prices.duplicated(price_keys).any():
        raise ValueError("Prices must be unique per store, item and M5 week.")
    if prices["sell_price"].dropna().le(0).any():
        raise ValueError("Known selling prices must be positive.")

    day_columns = sorted(
        (column for column in sales.columns if column.startswith("d_")),
        key=lambda column: int(column[2:]),
    )
    if not day_columns:
        raise ValueError("No daily sales columns found.")
    observed_calendar = calendar.set_index("d").reindex(day_columns)
    if observed_calendar["date"].isna().any():
        raise ValueError("Some sales days are missing from the calendar.")
    dates = pd.to_datetime(observed_calendar["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("Sales day numbers must follow calendar order.")

    day_weeks = dates + pd.to_timedelta(6 - dates.dt.dayofweek, unit="D")
    week_sizes = day_weeks.value_counts().sort_index()
    complete_weeks = week_sizes.index[week_sizes.eq(7)]
    if len(complete_weeks) == 0:
        raise ValueError("No complete calendar weeks available.")
    complete_columns = observed_calendar.index[day_weeks.isin(complete_weeks)].tolist()
    values = sales[complete_columns].to_numpy(copy=False)
    if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
        raise ValueError("Sales must be nonnegative integer counts.")
    weekly_values = values.reshape(len(sales), len(complete_weeks), 7).sum(axis=2)
    if weekly_values.max() <= np.iinfo(np.int32).max:
        weekly_values = weekly_values.astype("int32")

    observed_price_week_starts = observed_calendar.groupby("wm_yr_wk")["date"].min()
    known_prices = prices.loc[prices["sell_price"].notna(), price_keys].copy()
    known_prices["first_price_date"] = known_prices["wm_yr_wk"].map(observed_price_week_starts)
    first_prices = (
        known_prices.groupby(series_keys, observed=True)["first_price_date"]
        .min().reset_index()
    )
    series = sales[list(IDENTIFIERS)].merge(
        first_prices, on=series_keys, how="left", sort=False, validate="one_to_one"
    )
    weekly = pd.DataFrame({
        column: pd.Categorical(np.repeat(series[column].to_numpy(dtype=object), len(complete_weeks)))
        for column in IDENTIFIERS
    })
    weekly["week_end"] = np.tile(complete_weeks.to_numpy(), len(series))
    weekly["weekly_sales"] = weekly_values.reshape(-1)
    weekly["days_in_week"] = np.int8(7)
    weekly["first_price_date"] = np.repeat(series["first_price_date"].to_numpy(), len(complete_weeks))
    return weekly.sort_values([*series_keys, "week_end"]).reset_index(drop=True)

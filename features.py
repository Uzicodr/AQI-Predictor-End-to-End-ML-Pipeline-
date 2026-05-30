from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

POLLUTANTS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET = "aqi"
FORECAST_HORIZON = 72
TARGET_72H = "target_aqi_72h"


def build_feature_frame(
    df: pd.DataFrame,
    forecast_horizon: int = FORECAST_HORIZON,
    include_future_target: bool = True,
) -> pd.DataFrame:
    """Create reusable model features and the 72-hour-ahead target."""
    if df.empty:
        return df.copy()

    features = df.copy()
    features["datetime"] = pd.to_datetime(features["datetime"])
    features = features.sort_values("datetime").reset_index(drop=True)
    features = features.drop(columns=["_id"], errors="ignore")

    for col in POLLUTANTS + [TARGET]:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    dt = features["datetime"]
    features["hour"] = dt.dt.hour
    features["day"] = dt.dt.day
    features["month"] = dt.dt.month
    features["day_of_week"] = dt.dt.dayofweek
    features["day_of_year"] = dt.dt.dayofyear
    features["week"] = dt.dt.isocalendar().week.astype(int)

    features["pm10_lag1"] = features["pm10"].shift(1)
    features["pm10_lag3"] = features["pm10"].shift(3)
    features["pm2_5_lag1"] = features["pm2_5"].shift(1)
    features["aqi_lag1"] = features[TARGET].shift(1)
    features["co_lag12"] = features["co"].shift(12)

    features["pm_ratio"] = features["pm2_5"] / (features["pm10"] + 1e-6)
    features["pm2_5_x_no2"] = features["pm2_5"] * features["no2"]
    features["so2_x_pm10"] = features["so2"] * features["pm10"]

    features["aqi_change_rate"] = features[TARGET].diff()
    features["aqi_trend3"] = features[TARGET].diff(3)
    features["aqi_trend6"] = features[TARGET].diff(6)

    if include_future_target:
        features[TARGET_72H] = features[TARGET].shift(-forecast_horizon)

    return features.dropna().reset_index(drop=True)


def model_feature_columns(df: pd.DataFrame, target_col: str = TARGET_72H) -> list[str]:
    excluded = {"datetime", target_col}
    return [
        col
        for col in df.columns
        if col not in excluded and not col.startswith("_")
    ]


def write_csv(df: pd.DataFrame, path: Path, columns: Iterable[str] | None = None) -> None:
    path.parent.mkdir(exist_ok=True)
    if columns is not None:
        df = df[list(columns)]
    df.to_csv(path, index=False)

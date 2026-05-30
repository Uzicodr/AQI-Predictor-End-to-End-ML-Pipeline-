from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from pymongo import MongoClient, UpdateOne

from config import (
    AQI_DATA_FILE,
    DATA_DIR,
    LATITUDE,
    LONGITUDE,
    MONGODB_DB,
    MONGODB_URI,
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
)
from features import build_feature_frame, write_csv

RAW_COLLECTION = "raw_air_quality"
FEATURE_COLLECTION = "feature_store"
FEATURE_FILE = DATA_DIR / "aqi_features.csv"


def parse_date(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_openweather_history(start: datetime, end: datetime) -> list[dict]:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "appid": OPENWEATHER_API_KEY,
    }
    response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("list", [])


def raw_records_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in records:
        timestamp = entry.get("dt")
        components = entry.get("components", {})
        rows.append({
            "datetime": datetime.fromtimestamp(timestamp, timezone.utc).replace(tzinfo=None),
            "timestamp": timestamp,
            "aqi": entry.get("main", {}).get("aqi"),
            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
        })
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def upsert_frame(collection, df: pd.DataFrame, key: str = "datetime") -> int:
    if df.empty:
        return 0
    operations = []
    for record in df.to_dict("records"):
        operations.append(
            UpdateOne({key: record[key]}, {"$set": record}, upsert=True)
        )
    result = collection.bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count


def run_feature_pipeline(
    start: datetime,
    end: datetime,
    use_existing_csv: bool = False,
    skip_mongodb: bool = False,
) -> bool:
    print("AQI feature pipeline")
    print(f"Range: {start.isoformat()} to {end.isoformat()}")

    if use_existing_csv:
        raw_df = pd.read_csv(AQI_DATA_FILE)
    else:
        raw_records = fetch_openweather_history(start, end)
        raw_df = raw_records_to_frame(raw_records)
        if AQI_DATA_FILE.exists():
            existing_df = pd.read_csv(AQI_DATA_FILE)
            raw_df = pd.concat([existing_df, raw_df], ignore_index=True)

    if raw_df.empty:
        print("No raw records found")
        return False

    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
    raw_df = raw_df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    write_csv(raw_df, AQI_DATA_FILE)
    feature_df = build_feature_frame(raw_df, include_future_target=True)
    write_csv(feature_df, FEATURE_FILE)

    if skip_mongodb or not MONGODB_URI:
        print("MongoDB write skipped; wrote CSV files only")
        return True

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        db = client[MONGODB_DB]
        raw_count = upsert_frame(db[RAW_COLLECTION], raw_df)
        feature_count = upsert_frame(db[FEATURE_COLLECTION], feature_df)
        db[RAW_COLLECTION].create_index("datetime", unique=True)
        db[FEATURE_COLLECTION].create_index("datetime", unique=True)
        db[FEATURE_COLLECTION].create_index("target_aqi_72h")
        print(f"Stored raw records in MongoDB: {raw_count}")
        print(f"Stored engineered feature rows in MongoDB feature_store: {feature_count}")
        print(f"Feature CSV: {FEATURE_FILE}")
        return True
    except Exception as exc:
        print(f"Could not reach MongoDB; CSV artifacts were still written: {exc}")
        return True
    finally:
        try:
            client.close()
        except Exception:
            pass


def main() -> int:
    now = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser(description="Fetch AQI data, compute features, and store them.")
    parser.add_argument("--start", help="ISO start datetime, defaults to 5 days ago")
    parser.add_argument("--end", help="ISO end datetime, defaults to now")
    parser.add_argument("--use-existing-csv", action="store_true", help="Build feature store from data/aqi_data.csv")
    parser.add_argument("--skip-mongodb", action="store_true", help="Write CSV artifacts without connecting to MongoDB")
    args = parser.parse_args()

    start = parse_date(args.start, now - timedelta(days=5))
    end = parse_date(args.end, now)
    return 0 if run_feature_pipeline(start, end, args.use_existing_csv, args.skip_mongodb) else 1


if __name__ == "__main__":
    sys.exit(main())

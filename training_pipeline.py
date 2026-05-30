import pickle
import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import AQI_DATA_FILE, DATA_DIR, MONGODB_DB, MONGODB_URI
from feature_pipeline import FEATURE_FILE
from features import TARGET_72H, build_feature_frame, model_feature_columns

RANDOM_STATE = 42
HOLDOUT_HOURS = 72
MODEL_COLLECTION = "model_registry"


def load_training_frame() -> pd.DataFrame:
    if MONGODB_URI and os.getenv("SKIP_MONGODB") != "1":
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
            client.admin.command("ping")
            records = list(
                client[MONGODB_DB]["feature_store"]
                .find({TARGET_72H: {"$exists": True}}, {"_id": 0})
                .sort("datetime", 1)
            )
            client.close()
            if records:
                print(f"Loaded {len(records)} feature rows from MongoDB feature_store")
                return pd.DataFrame(records)
        except Exception as exc:
            print(f"Feature store unavailable, falling back to local files: {exc}")

    if FEATURE_FILE.exists():
        print(f"Loaded features from {FEATURE_FILE}")
        return pd.read_csv(FEATURE_FILE)

    if not AQI_DATA_FILE.exists():
        raise FileNotFoundError(f"No feature store or CSV data found. Missing {AQI_DATA_FILE}")

    print(f"Building features from {AQI_DATA_FILE}")
    return build_feature_frame(pd.read_csv(AQI_DATA_FILE), include_future_target=True)


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmse = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    ).mean()

    return {
        "name": name,
        "model": model,
        "predictions": pred,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "cv_rmse": float(cv_rmse),
    }


def save_model_registry_entry(metrics: dict, feature_cols: list[str], artifact_paths: dict) -> None:
    registry_doc = {
        "model_name": "aqi_72h_forecaster",
        "model_type": metrics["name"],
        "created_at": datetime.now(timezone.utc),
        "target": TARGET_72H,
        "forecast_horizon_hours": HOLDOUT_HOURS,
        "metrics": {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "cv_rmse": metrics["cv_rmse"],
        },
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
    }

    DATA_DIR.mkdir(exist_ok=True)
    registry_file = DATA_DIR / "model_registry_latest.pkl"
    with open(registry_file, "wb") as f:
        pickle.dump(registry_doc, f)

    if not MONGODB_URI or os.getenv("SKIP_MONGODB") == "1":
        print(f"MongoDB not configured; registry metadata saved locally to {registry_file}")
        return

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        collection = client[MONGODB_DB][MODEL_COLLECTION]
        collection.insert_one(registry_doc)
        collection.create_index("created_at")
        print(f"Registered model in MongoDB collection: {MODEL_COLLECTION}")
    except Exception as exc:
        print(f"Could not write MongoDB model registry entry: {exc}")
    finally:
        try:
            client.close()
        except Exception:
            pass


def train_model() -> bool:
    print("=" * 70)
    print("AQI 72-hour Forecast Training Pipeline")
    print("=" * 70)

    try:
        df = load_training_frame()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").dropna().reset_index(drop=True)

        feature_cols = model_feature_columns(df, TARGET_72H)
        if len(df) <= HOLDOUT_HOURS:
            raise ValueError("Not enough rows for a 72-hour holdout split")

        split_idx = len(df) - HOLDOUT_HOURS
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_72H].astype(float)
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_72H].astype(float)

        print(f"Rows: {len(df)} | Train: {len(train_df)} | 72h holdout: {len(test_df)}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Features: {len(feature_cols)}")

        models = [
            ("Ridge Regression", Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])),
            ("Random Forest", RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
            ("XGBoost", xgb.XGBRegressor(
                n_estimators=700,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )),
        ]

        results = [
            evaluate_model(name, model, X_train, y_train, X_test, y_test)
            for name, model in models
        ]
        results = sorted(results, key=lambda item: item["rmse"])

        print("\nModel comparison")
        for result in results:
            print(
                f"{result['name']:<18} "
                f"RMSE={result['rmse']:.4f} "
                f"MAE={result['mae']:.4f} "
                f"R2={result['r2']:.4f} "
                f"CV_RMSE={result['cv_rmse']:.4f}"
            )

        best = results[0]
        final_model = best["model"]
        final_model.fit(df[feature_cols], df[TARGET_72H].astype(float))

        shap_importance = pd.DataFrame()
        if best["name"] in {"Random Forest", "XGBoost"}:
            sample = df[feature_cols].tail(min(250, len(df)))
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(sample)
            shap_importance = pd.DataFrame({
                "feature": feature_cols,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
        else:
            coefficients = final_model.named_steps["ridge"].coef_
            shap_importance = pd.DataFrame({
                "feature": feature_cols,
                "mean_abs_shap": np.abs(coefficients),
            }).sort_values("mean_abs_shap", ascending=False)

        DATA_DIR.mkdir(exist_ok=True)
        model_path = DATA_DIR / "aqi_xgb_model.pkl"
        forecast_model_path = DATA_DIR / "aqi_forecast_model.pkl"
        le_path = DATA_DIR / "label_encoder.pkl"
        feat_path = DATA_DIR / "feature_columns.pkl"
        shap_path = DATA_DIR / "shap_importance.pkl"
        shap_csv_path = DATA_DIR / "shap_feature_importance.csv"

        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        with open(forecast_model_path, "wb") as f:
            pickle.dump(final_model, f)
        with open(le_path, "wb") as f:
            pickle.dump(None, f)
        with open(feat_path, "wb") as f:
            pickle.dump({
                "feature_cols": feature_cols,
                "selected_features": feature_cols,
                "target": TARGET_72H,
                "forecast_horizon_hours": HOLDOUT_HOURS,
                "model_type": best["name"],
            }, f)
        with open(shap_path, "wb") as f:
            pickle.dump(shap_importance, f)
        shap_importance.to_csv(shap_csv_path, index=False)

        save_model_registry_entry(
            best,
            feature_cols,
            {
                "model": model_path,
                "forecast_model": forecast_model_path,
                "features": feat_path,
                "importance": shap_csv_path,
            },
        )

        print("\nFINAL RESULTS")
        print(f"Best model:     {best['name']}")
        print(f"RMSE:           {best['rmse']:.4f}")
        print(f"MAE:            {best['mae']:.4f}")
        print(f"R2:             {best['r2']:.4f}")
        print(f"Model saved:    {forecast_model_path}")
        print(f"App alias:      {model_path}")
        print(f"SHAP saved:     {shap_csv_path}")
        return True

    except Exception as exc:
        print(f"Error during training: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)

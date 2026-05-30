# Pearls AQI Predictor - Project Report

## Requirement Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Fetch raw weather/pollutant data from an external API | Done | `feature_pipeline.py` and `app.py` use OpenWeather air pollution endpoints. |
| Compute model features and targets | Done | `features.py` creates time features, pollutant interactions, AQI lags/trends, `aqi_change_rate`, and `target_aqi_72h`. |
| Store features in a feature store | Done with MongoDB Atlas | MongoDB `feature_store` collection is used instead of Hopsworks/Vertex AI. |
| Backfill historical features and targets | Done | `backfill.py` builds engineered rows from `data/aqi_data.csv` and writes them to MongoDB. |
| Train and evaluate ML models | Done | `training_pipeline.py` compares Ridge Regression, Random Forest, and XGBoost. |
| Evaluate with RMSE, MAE, and R2 | Done | `training_pipeline.py` prints and stores all three metrics. |
| Store trained model in a model registry | Done with MongoDB Atlas | `training_pipeline.py` writes metadata to MongoDB `model_registry` and local `data/model_registry_latest.pkl`. |
| Automate feature script hourly | Done | `.github/workflows/train.yml` runs `feature_pipeline.py` every hour. |
| Automate training script daily | Done | `.github/workflows/train.yml` runs `training_pipeline.py` daily at 00:15 UTC. |
| Dashboard with predictions | Done | `app.py` provides a Streamlit dashboard with live AQI, 72-hour outlook, history, pollutant cards, and alerts. |
| Load model and features from feature store | Done | `app.py` loads recent data from MongoDB `feature_store` and model artifacts from `data/`. |
| EDA to identify trends | Partially done | Dashboard trend charts and SHAP importance are present; notebook exists in `model_training/`. |
| SHAP/LIME explanations | Done | SHAP feature importance is saved to `data/shap_feature_importance.csv`. |
| Hazardous AQI alerts | Done | `app.py` displays a health alert when AQI exceeds unhealthy thresholds. |
| Deep learning model | Not implemented | TensorFlow/PyTorch is not included because the dataset is small and the final pipeline focuses on stable classical models plus XGBoost. |

## What Was Built

This project is an end-to-end AQI forecasting system for Karachi using a fully serverless deployment style:

- OpenWeather supplies pollutant and weather data.
- MongoDB Atlas is used as the feature store and model registry.
- GitHub Actions automates hourly feature generation and daily model training.
- Streamlit serves the user-facing dashboard.
- XGBoost/Random Forest/Ridge Regression are compared for 72-hour-ahead AQI prediction.

## What Was Not Built

- Hopsworks or Vertex AI were not used. MongoDB Atlas replaces them as the feature store and registry.
- A TensorFlow/PyTorch deep learning model was not added. The current data volume is better suited to Ridge, Random Forest, and XGBoost.
- The dashboard uses Streamlit only; Flask/FastAPI is not needed unless an API layer is required separately.

## How To Run

```bash
python feature_pipeline.py
python backfill.py
python training_pipeline.py
streamlit run app.py
```

Required environment variables:

```bash
OPENWEATHER_API_KEY=...
MONGODB_URI=...
```

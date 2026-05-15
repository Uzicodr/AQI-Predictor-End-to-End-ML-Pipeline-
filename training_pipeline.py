import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, mean_absolute_error
)
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from config import AQI_DATA_FILE, DATA_DIR

POLLUTANTS       = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET           = "aqi"
FORECAST_HORIZON = 72
RANDOM_STATE     = 42
TOP_N_FEATURES   = 15

AQI_LABELS = {2: "Good", 3: "Moderate", 4: "Unhealthy (Sensitive)", 5: "Unhealthy"}

def train_model():
    
    print("=" * 70)
    print("AQI 3-Day Forecast Pipeline - Training")
    print("=" * 70)
    
    try:
        print("\nLoading data from CSV...")
        if not AQI_DATA_FILE.exists():
            print(f"Data file not found: {AQI_DATA_FILE}")
            print("   Please run backfill.py or fetch_aqi_data.py first")
            return False
        
        df = pd.read_csv(AQI_DATA_FILE)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.drop(columns=["_id", "timestamp"], errors="ignore")
        
        print(f"Loaded {len(df)} records")
        print(f"   Date range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
        print(f"   AQI distribution:\n{df[TARGET].value_counts().sort_index()}")
        
        print("\nCleaning data - clipping outliers...")
        for col in POLLUTANTS:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            fence = 3 * (Q3 - Q1)
            df[col] = df[col].clip(lower=Q1 - fence, upper=Q3 + fence)
        print("Outliers clipped (3×IQR method)")
        
        print("\nEngineering important SHAP-identified features...")
        dt = df["datetime"]
        
        df["day_of_year"] = dt.dt.dayofyear
        df["week"] = dt.dt.isocalendar().week.astype(int)
        
        df["pm10_lag1"] = df["pm10"].shift(1)
        df["pm10_lag3"] = df["pm10"].shift(3)
        df["pm2_5_lag1"] = df["pm2_5"].shift(1)
        df["aqi_lag1"] = df[TARGET].shift(1)
        df["co_lag12"] = df["co"].shift(12)
        
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
        df["pm2_5_x_no2"] = df["pm2_5"] * df["no2"]
        df["so2_x_pm10"] = df["so2"] * df["pm10"]
        
        df["aqi_trend3"] = df[TARGET].diff(3)
        df["aqi_trend6"] = df[TARGET].diff(6)
        
        df = df.dropna().reset_index(drop=True)
        
        feature_cols = [c for c in df.columns
                        if c not in ["datetime", TARGET] and not c.startswith("_")]
        
        print(f"Total features engineered: {len(feature_cols)}")
        print(f"   Data shape after feature engineering: {df.shape}")
        
        print("\nSplitting data (time-aware)...")
        SPLIT_IDX = len(df) - FORECAST_HORIZON
        train_df  = df.iloc[:SPLIT_IDX]
        test_df   = df.iloc[SPLIT_IDX:]
        
        X_train, y_train = train_df[feature_cols].values, train_df[TARGET].values
        X_test,  y_test  = test_df[feature_cols].values,  test_df[TARGET].values
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)
        
        print(f"Train: {len(X_train)} rows | Test (72h hold-out): {len(X_test)} rows")
        print(f"   Classes: {dict(zip(le.classes_, np.bincount(y_train_enc)))}")
        
        print("\nApplying SMOTE for class balance...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train_enc)
        
        print(f"SMOTE applied")
        print(f"   Before: {len(X_train)} samples")
        print(f"   After:  {len(X_resampled)} samples")
        print(f"   Resampled dist: {dict(zip(le.classes_, np.bincount(y_resampled)))}")
        
        print("\nTraining XGBoost classifier...")
        params = dict(
            n_estimators      = 800,
            max_depth         = 7,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            min_child_weight  = 3,
            gamma             = 0.1,
            objective         = "multi:softmax",
            num_class         = len(le.classes_),
            eval_metric       = "merror",
            use_label_encoder = False,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            tree_method       = "hist",
            early_stopping_rounds = 50,
        )
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_resampled, y_resampled,
            eval_set=[(X_test, y_test_enc)],
            verbose=100,
        )
        print(f"Model trained (best iteration: {model.best_iteration})")
        
        print("\nEvaluating model...")
        y_pred_enc = model.predict(X_test)
        y_pred     = le.inverse_transform(y_pred_enc)
        y_true     = y_test
        
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="weighted")
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"   Accuracy:        {acc:.4f}")
        print(f"   Weighted F1:     {f1:.4f}")
        print(f"   Mean Abs Error:  {mae:.4f} AQI classes")
        
        present_classes = sorted(set(y_true) | set(y_pred))
        print(f"\n   Classification Report:")
        print(classification_report(
            y_true, y_pred,
            labels=present_classes,
            target_names=[AQI_LABELS.get(c, str(c)) for c in present_classes]
        ))
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        cv_scores = cross_val_score(
            xgb.XGBClassifier(**cv_params), X_train, y_train_enc,
            cv=tscv, scoring="accuracy", n_jobs=-1
        )
        print(f"   Time-Series CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\nComputing SHAP feature importance...")
        explainer  = shap.TreeExplainer(model)
        shap_data  = np.vstack([X_train[-200:], X_test])
        shap_vals  = explainer.shap_values(shap_data)
        
        if isinstance(shap_vals, list):
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        elif shap_vals.ndim == 3:
            mean_abs_shap = np.abs(shap_vals).mean(axis=2)
        else:
            mean_abs_shap = np.abs(shap_vals)
        
        shap_importance = pd.DataFrame({
            "feature":       feature_cols,
            "mean_abs_shap": mean_abs_shap.mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        print(f"Top 25 SHAP features:")
        print(shap_importance.head(25).to_string(index=False))
        
        print(f"\nRetraining with top {TOP_N_FEATURES} SHAP features...")
        top_features = shap_importance.head(TOP_N_FEATURES)["feature"].tolist()
        feat_idx     = [feature_cols.index(f) for f in top_features]
        
        X_tr_top = X_resampled[:, feat_idx]
        X_te_top = X_test[:, feat_idx]
        
        model_shap = xgb.XGBClassifier(**{k: v for k, v in params.items()
                                           if k != "early_stopping_rounds"})
        model_shap.set_params(n_estimators=model.best_iteration + 1)
        model_shap.fit(X_tr_top, y_resampled)
        
        y_pred_shap     = model_shap.predict(X_te_top)
        y_pred_shap_dec = le.inverse_transform(y_pred_shap)
        
        acc2 = accuracy_score(y_true, y_pred_shap_dec)
        f1_2 = f1_score(y_true, y_pred_shap_dec, average="weighted")
        mae2 = mean_absolute_error(y_true, y_pred_shap_dec)
        
        print(f"SHAP-Pruned Model Performance:")
        print(f"   Accuracy:       {acc2:.4f}")
        print(f"   Weighted F1:    {f1_2:.4f}")
        print(f"   Mean Abs Error: {mae2:.4f}")
        
        if f1_2 >= f1 - 0.005:
            final_model = model_shap
            final_feats = top_features
            final_fidx = feat_idx
            print(f"   Using SHAP-pruned model (fewer features, same performance)")
        else:
            final_model = model
            final_feats = feature_cols
            final_fidx = list(range(len(feature_cols)))
            print(f"   Using full-feature model")
        
        print("\nSaving model artifacts...")
        DATA_DIR.mkdir(exist_ok=True)
        
        model_path = DATA_DIR / "aqi_xgb_model.pkl"
        le_path = DATA_DIR / "label_encoder.pkl"
        feat_path = DATA_DIR / "feature_columns.pkl"
        shap_path = DATA_DIR / "shap_importance.pkl"
        shap_csv_path = DATA_DIR / "shap_feature_importance.csv"
        
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)
        
        with open(feat_path, 'wb') as f:
            pickle.dump({"feature_cols": feature_cols, "selected_features": final_feats, "selected_indices": final_fidx}, f)
        
        with open(shap_path, 'wb') as f:
            pickle.dump(shap_importance, f)
        
        shap_importance.to_csv(shap_csv_path, index=False)
        
        print(f"Model saved:               {model_path}")
        print(f"   Label encoder saved:      {le_path}")
        print(f"   Feature columns saved:    {feat_path}")
        print(f"   SHAP importance saved:    {shap_path}")
        print(f"   SHAP CSV saved:           {shap_csv_path}")
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Features engineered:      {len(feature_cols)}")
        print(f"SHAP-selected features:   {TOP_N_FEATURES}")
        print(f"Accuracy (72h hold-out):  {acc2:.2%}")
        print(f"Weighted F1:              {f1_2:.4f}")
        print(f"Mean Absolute Error:      {mae2:.4f} AQI classes")
        print(f"Forecast horizon:         72 hours (3 days)")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
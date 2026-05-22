"""
app.py — Karachi AQI Predictor Dashboard
Run locally:  streamlit run app.py
Deploy:       streamlit.io/cloud
"""

import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient
import certifi

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "aqi_predictor")
MODEL_PATH = Path("data/aqi_xgb_model.pkl")
LE_PATH = Path("data/label_encoder.pkl")
FEAT_PATH = Path("data/feature_columns.pkl")
POLLUTANTS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET = "aqi"

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main { background: #0a0e1a; }
.stApp { background: #0a0e1a; }

.aqi-card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s;
}
.aqi-card:hover { transform: translateY(-2px); }

.aqi-number {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
}

.aqi-label {
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}

.aqi-category {
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 8px;
}

.alert-box {
    background: linear-gradient(135deg, #3b0000, #1a0000);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 16px 20px;
    color: #fca5a5;
    font-weight: 500;
    margin: 16px 0;
}

.pollutant-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #e5e7eb;
    margin-bottom: 16px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


def aqi_info(val):
    if val is None:
        return "Unknown", "#6b7280", "❓"
    val = float(val)
    if val <= 50:   return "Good",                     "#22c55e", "🟢"
    if val <= 100:  return "Moderate",                 "#eab308", "🟡"
    if val <= 150:  return "Unhealthy for Sensitive",  "#f97316", "🟠"
    if val <= 200:  return "Unhealthy",                "#ef4444", "🔴"
    if val <= 300:  return "Very Unhealthy",           "#a855f7", "🟣"
    return              "Hazardous",                   "#dc2626", "⚫"


def openweather_aqi_info(val):
    if val is None:
        return "Unknown", "#6b7280", "?"
    labels = {
        1: ("Good", "#22c55e", "1"),
        2: ("Fair", "#84cc16", "2"),
        3: ("Moderate", "#eab308", "3"),
        4: ("Poor", "#f97316", "4"),
        5: ("Very Poor", "#ef4444", "5"),
    }
    return labels.get(int(round(float(val))), ("Unknown", "#6b7280", "?"))


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists() or not LE_PATH.exists() or not FEAT_PATH.exists():
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    with open(FEAT_PATH, "rb") as f:
        feat_info = pickle.load(f)
    return model, le, feat_info


@st.cache_data(ttl=3600, show_spinner="Fetching live AQI...")
def fetch_live():
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat=24.8607&lon=67.0011&appid={OPENWEATHER_API_KEY}"
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat=24.8607&lon=67.0011&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        aqi_resp = requests.get(url, timeout=10).json()
        weather_resp = requests.get(weather_url, timeout=10).json()
    except Exception:
        return None
    
    if aqi_resp.get("list") is None or weather_resp.get("main") is None:
        return None
    
    aqi_data = aqi_resp["list"][0]
    components = aqi_data.get("components", {})
    main = weather_resp.get("main", {})
    wind = weather_resp.get("wind", {})
    
    def calculate_aqi(components):
        pm2_5 = components.get("pm2_5")
        pm10 = components.get("pm10")
        no2 = components.get("no2")
        o3 = components.get("o3")
        
        aqi_values = []
        
        if pm2_5 is not None:
            if pm2_5 <= 12:
                aqi_values.append((pm2_5 / 12) * 50)
            elif pm2_5 <= 35.4:
                aqi_values.append(50 + ((pm2_5 - 12) / 23.4) * 50)
            elif pm2_5 <= 55.4:
                aqi_values.append(100 + ((pm2_5 - 35.4) / 20) * 50)
            elif pm2_5 <= 150.4:
                aqi_values.append(150 + ((pm2_5 - 55.4) / 95) * 50)
            else:
                aqi_values.append(200 + ((pm2_5 - 150.4) / 249.6) * 100)
        
        if pm10 is not None:
            if pm10 <= 54:
                aqi_values.append((pm10 / 54) * 50)
            elif pm10 <= 154:
                aqi_values.append(50 + ((pm10 - 54) / 100) * 50)
            elif pm10 <= 254:
                aqi_values.append(100 + ((pm10 - 154) / 100) * 50)
            elif pm10 <= 354:
                aqi_values.append(150 + ((pm10 - 254) / 100) * 50)
            else:
                aqi_values.append(200 + ((pm10 - 354) / 646) * 100)
        
        if no2 is not None:
            if no2 <= 53:
                aqi_values.append((no2 / 53) * 50)
            elif no2 <= 100:
                aqi_values.append(50 + ((no2 - 53) / 47) * 50)
            elif no2 <= 360:
                aqi_values.append(100 + ((no2 - 100) / 260) * 50)
            elif no2 <= 649:
                aqi_values.append(150 + ((no2 - 360) / 289) * 50)
            else:
                aqi_values.append(200 + ((no2 - 649) / 1251) * 100)
        
        if o3 is not None:
            if o3 <= 54:
                aqi_values.append((o3 / 54) * 50)
            elif o3 <= 70:
                aqi_values.append(50 + ((o3 - 54) / 16) * 50)
            elif o3 <= 85:
                aqi_values.append(100 + ((o3 - 70) / 15) * 50)
            elif o3 <= 105:
                aqi_values.append(150 + ((o3 - 85) / 20) * 50)
            else:
                aqi_values.append(200 + ((o3 - 105) / 405) * 100)
        
        return max(aqi_values) if aqi_values else 50
    
    aqi = calculate_aqi(components)
    
    return {
        "aqi"        : aqi,
        "pm2_5"      : components.get("pm2_5"),
        "pm10"       : components.get("pm10"),
        "no2"        : components.get("no2"),
        "o3"         : components.get("o3"),
        "co"         : components.get("co"),
        "so2"        : components.get("so2"),
        "no"         : components.get("no"),
        "nh3"        : components.get("nh3"),
        "temperature": main.get("temp"),
        "humidity"   : main.get("humidity"),
        "wind_speed" : wind.get("speed"),
        "pressure"   : main.get("pressure"),
        "time"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "station"    : "Karachi",
    }


@st.cache_data(ttl=3600, show_spinner="Loading history...")
def load_history():
    if not MONGODB_URI:
        return pd.DataFrame()
    try:
        client = MongoClient(MONGODB_URI, tls=True, tlsCAFile=certifi.where(),
                             serverSelectionTimeoutMS=8000)
        db  = client[MONGODB_DB]
        col = db["aqi_data"]
        records = list(col.find({}, {"_id": 0}).sort("datetime", -1).limit(720))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime")
    except Exception:
        return pd.DataFrame()


def predict_72h(model, le, feat_info, live):
    predictions = []
    if model is None or live is None or le is None:
        return predictions
    
    feature_cols = feat_info["feature_cols"]
    df_ext = pd.DataFrame([live])
    last_time = datetime.now()
    
    for h in range(1, 73):
        next_time = last_time + timedelta(hours=h)
        row = {}
        
        row["day_of_year"] = next_time.timetuple().tm_yday
        row["week"] = next_time.isocalendar()[1]
        
        row["pm10_lag1"] = df_ext["pm10"].iloc[-1] if "pm10" in df_ext.columns else 0
        row["pm10_lag3"] = df_ext["pm10"].iloc[-1] if "pm10" in df_ext.columns else 0
        row["pm2_5_lag1"] = df_ext["pm2_5"].iloc[-1] if "pm2_5" in df_ext.columns else 0
        row["aqi_lag1"] = df_ext[TARGET].iloc[-1] if TARGET in df_ext.columns else 3
        row["co_lag12"] = df_ext["co"].iloc[-1] if "co" in df_ext.columns else 0
        
        pm10_val = df_ext["pm10"].iloc[-1] if "pm10" in df_ext.columns else 50
        pm2_5_val = df_ext["pm2_5"].iloc[-1] if "pm2_5" in df_ext.columns else 30
        no2_val = df_ext["no2"].iloc[-1] if "no2" in df_ext.columns else 30
        so2_val = df_ext["so2"].iloc[-1] if "so2" in df_ext.columns else 10
        
        row["pm_ratio"] = pm2_5_val / (pm10_val + 1e-6)
        row["pm2_5_x_no2"] = pm2_5_val * no2_val
        row["so2_x_pm10"] = so2_val * pm10_val
        
        row["aqi_trend3"] = 0
        row["aqi_trend6"] = 0
        
        for col in POLLUTANTS:
            if col not in row:
                row[col] = df_ext[col].iloc[-1] if col in df_ext.columns else 0
        
        x_vec = np.array([row.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
        
        try:
            pred_enc = model.predict(x_vec)[0]
            pred_proba = model.predict_proba(x_vec)[0]
            pred_aqi = int(le.inverse_transform([pred_enc])[0])
            confidence = float(pred_proba.max())
            
            predictions.append({
                "datetime": next_time,
                "hour": next_time.strftime("%H:%M"),
                "day": next_time.strftime("%a"),
                "date": next_time.strftime("%b %d"),
                "predicted_aqi": pred_aqi,
                "confidence": round(confidence * 100, 1),
            })
            
            new_row = {c: np.nan for c in POLLUTANTS + [TARGET]}
            new_row[TARGET] = pred_aqi
            for col in POLLUTANTS:
                new_row[col] = row.get(col, 0)
            df_ext = pd.concat([df_ext, pd.DataFrame([new_row])], ignore_index=True)
            
        except Exception as e:
            st.warning(f"Prediction error at hour {h}: {e}")
            break
    
    return predictions


model, le, feat_info = load_model()
live = fetch_live()
history = load_history()
predictions = predict_72h(model, le, feat_info, live)

st.markdown("""
<div style='padding: 8px 0 24px 0;'>
  <div style='font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #f9fafb;'>
    🌫️ Karachi AQI Predictor
  </div>
  <div style='color: #6b7280; font-size: 0.9rem; margin-top: 4px;'>
    Real-time air quality monitoring & 3-day forecast
  </div>
</div>
""", unsafe_allow_html=True)

if live is None:
    st.error("Could not fetch live data from OpenWeather. Check your API key.")
    st.stop()

current_aqi = live["aqi"]
cat, color, emoji = aqi_info(current_aqi)

if current_aqi > 150:
    st.markdown(f"""
    <div class='alert-box'>
        ⚠️ <strong>HEALTH ALERT</strong> — Current AQI is <strong>{current_aqi:.0f}</strong> ({cat}).
        Avoid prolonged outdoor activity. Sensitive groups should stay indoors.
    </div>
    """, unsafe_allow_html=True)

cols = st.columns(4)

with cols[0]:
    st.markdown(f"""
    <div class='aqi-card' style='border-color: {color}40;'>
        <div class='aqi-label'>Right Now</div>
        <div class='aqi-number' style='color: {color};'>{current_aqi:.0f}</div>
        <div class='aqi-category' style='color: {color};'>{emoji} {cat}</div>
        <div style='color: #6b7280; font-size: 0.75rem; margin-top: 8px;'>{live["time"]}</div>
    </div>
    """, unsafe_allow_html=True)

if predictions:
    for i, pred in enumerate(predictions[::24][:3]):
        pc, pcolor, pclass = openweather_aqi_info(pred["predicted_aqi"])
        with cols[i + 1]:
            st.markdown(f"""
            <div class='aqi-card' style='border-color: {pcolor}40;'>
                <div class='aqi-label'>{pred["day"]}</div>
                <div style='font-family: Syne, sans-serif; font-size: 1.65rem; font-weight: 800; color: {pcolor}; line-height: 1.1;'>{pc}</div>
                <div class='aqi-category' style='color: {pcolor};'>OpenWeather class {pclass}</div>
                <div style='color: #6b7280; font-size: 0.75rem; margin-top: 8px;'>{pred["date"]}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("<div class='section-title'>📈 Historical AQI Trend</div>", unsafe_allow_html=True)
    if not history.empty:
        hist7 = history.tail(168)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist7["datetime"], y=hist7["aqi"],
            mode="lines", name="AQI",
            line=dict(color="#60a5fa", width=2),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)"
        ))
        fig.add_hline(y=100, line_dash="dot", line_color="#eab308",
                      annotation_text="Moderate", annotation_font_color="#eab308")
        fig.add_hline(y=150, line_dash="dot", line_color="#ef4444",
                      annotation_text="Unhealthy", annotation_font_color="#ef4444")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data yet. Run backfill.py first.")

with col_right:
    st.markdown("<div class='section-title'>📊 3-Day Forecast</div>", unsafe_allow_html=True)
    if predictions:
        forecast_points = predictions[::6][:12]
        hours = [p["hour"] for p in forecast_points]
        aqi_v = [p["predicted_aqi"] for p in forecast_points]
        labels = [openweather_aqi_info(v)[0] for v in aqi_v]
        colors_list = [openweather_aqi_info(v)[1] for v in aqi_v]

        fig2 = go.Figure(go.Bar(
            x=hours, y=aqi_v,
            marker_color=colors_list,
            text=labels,
            textposition="outside",
            textfont=dict(color="#e5e7eb", size=13)
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af",
            height=280,
            margin=dict(l=0, r=0, t=24, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                title="OpenWeather AQI class",
                gridcolor="rgba(255,255,255,0.05)",
                range=[0, 5.8],
                tickmode="array",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Good", "Fair", "Moderate", "Poor", "Very Poor"],
            ),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div class='section-title'>🔬 Pollutant Breakdown</div>", unsafe_allow_html=True)

pollutants = {
    "PM2.5": (live.get("pm2_5"), "µg/m³", 35),
    "PM10" : (live.get("pm10"),  "µg/m³", 50),
    "NO₂"  : (live.get("no2"),   "µg/m³", 25),
    "O₃"   : (live.get("o3"),    "µg/m³", 60),
    "CO"   : (live.get("co"),    "µg/m³", 4000),
    "SO₂"  : (live.get("so2"),   "µg/m³", 20),
}

pcols = st.columns(len(pollutants))
for col, (name, (val, unit, safe_limit)) in zip(pcols, pollutants.items()):
    with col:
        if val is not None:
            pct     = min(float(val) / safe_limit, 1.0)
            pcolor  = "#22c55e" if pct < 0.5 else "#eab308" if pct < 0.8 else "#ef4444"
            st.markdown(f"""
            <div class='aqi-card' style='padding: 16px;'>
                <div style='color: #6b7280; font-size: 0.75rem; letter-spacing: 0.1em; 
                            text-transform: uppercase;'>{name}</div>
                <div style='font-family: Syne, sans-serif; font-size: 1.6rem; 
                            font-weight: 700; color: {pcolor}; margin: 4px 0;'>
                    {val:.1f}
                </div>
                <div style='color: #6b7280; font-size: 0.72rem;'>{unit}</div>
                <div style='background: #1f2937; border-radius: 4px; 
                            height: 4px; margin-top: 10px;'>
                    <div style='background: {pcolor}; width: {pct*100:.0f}%; 
                                height: 4px; border-radius: 4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='aqi-card' style='padding: 16px;'>
                <div style='color: #6b7280; font-size: 0.75rem; letter-spacing: 0.1em;
                            text-transform: uppercase;'>{name}</div>
                <div style='color: #374151; font-size: 1.2rem; margin-top: 8px;'>N/A</div>
            </div>
            """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🌤 Weather")
    weather = {
        "🌡 Temperature" : f"{live['temperature']} °C" if live['temperature'] else "N/A",
        "💧 Humidity"    : f"{live['humidity']} %"     if live['humidity']    else "N/A",
        "💨 Wind"        : f"{live['wind_speed']} m/s" if live['wind_speed']  else "N/A",
        "🔽 Pressure"    : f"{live['pressure']} hPa"   if live['pressure']    else "N/A",
    }
    for label, val in weather.items():
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; 
                    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
                    color: #d1d5db; font-size: 0.9rem;'>
            <span>{label}</span><span style='color: #60a5fa;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("###Model Info")
    if model is not None:
        st.success("Model loaded ✓")
        st.caption("Type: XGBoost Classifier")
        feature_cols = (feat_info.get("selected_features") or feat_info.get("feature_cols") or []) if feat_info else []
        st.caption(f"Features: {len(feature_cols)}")
        if not history.empty:
            st.caption(f"Trained on: {len(history)} records")
    else:
        st.warning("Model not found. Run training_pipeline.py first.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📡 Station")
    st.caption(live.get("station", "Karachi"))
    st.caption("Source: OpenWeather API")
    st.caption(f"Updated: {live['time']}")

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

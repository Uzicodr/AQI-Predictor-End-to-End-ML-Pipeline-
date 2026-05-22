"""
Karachi AQI Predictor Dashboard

Run locally:
    streamlit run app.py
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "aqi_predictor")
MODEL_PATH = Path("data/aqi_xgb_model.pkl")
LE_PATH = Path("data/label_encoder.pkl")
FEAT_PATH = Path("data/feature_columns.pkl")
POLLUTANTS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
TARGET = "aqi"

st.set_page_config(
    page_title="Karachi AQI Monitor",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1220;
    --panel: #111827;
    --panel-soft: #162033;
    --border: rgba(148, 163, 184, 0.22);
    --muted: #94a3b8;
    --text: #e5eefb;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(20, 184, 166, 0.12), transparent 30%),
        linear-gradient(180deg, #0b1220 0%, #111827 58%, #0b1220 100%);
}

section[data-testid="stSidebar"] {
    background: rgba(11, 18, 32, 0.92);
    border-right: 1px solid var(--border);
}

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 2rem;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 20px;
    margin-bottom: 22px;
}

.eyebrow {
    color: #38bdf8;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.title {
    color: var(--text);
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 800;
    letter-spacing: 0;
    line-height: 1.02;
    margin-top: 4px;
}

.subtitle {
    color: var(--muted);
    margin-top: 9px;
    max-width: 760px;
}

.status-pill {
    border: 1px solid var(--border);
    background: rgba(17, 24, 39, 0.72);
    color: #cbd5e1;
    border-radius: 999px;
    padding: 9px 13px;
    font-size: 0.84rem;
    white-space: nowrap;
}

.metric-card, .forecast-card, .panel, .pollutant-card {
    background: rgba(17, 24, 39, 0.84);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.20);
}

.metric-card {
    min-height: 228px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.label {
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.aqi-value {
    font-size: clamp(3.6rem, 7vw, 6.2rem);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: 0;
}

.category {
    font-size: 1.18rem;
    font-weight: 800;
    margin-top: 8px;
}

.fineprint {
    color: var(--muted);
    font-size: 0.8rem;
}

.forecast-card {
    min-height: 228px;
    padding: 20px;
}

.forecast-date {
    color: #cbd5e1;
    font-size: 0.88rem;
    margin-top: 2px;
}

.forecast-value {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    margin-top: 25px;
}

.forecast-note {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 14px;
}

.panel {
    padding: 18px;
    min-height: 340px;
}

.section-title {
    color: #e2e8f0;
    font-size: 0.92rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.pollutant-card {
    padding: 16px;
    min-height: 128px;
}

.pollutant-value {
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 6px;
}

.bar-track {
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.18);
    overflow: hidden;
    margin-top: 13px;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.alert-box {
    background: rgba(127, 29, 29, 0.32);
    border: 1px solid rgba(248, 113, 113, 0.55);
    color: #fecaca;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 18px;
    font-weight: 600;
}

@media (max-width: 760px) {
    .topbar {
        align-items: flex-start;
        flex-direction: column;
    }
    .status-pill {
        white-space: normal;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def aqi_info(value):
    if value is None or pd.isna(value):
        return "Unknown", "#94a3b8"
    value = float(value)
    if value <= 50:
        return "Good", "#22c55e"
    if value <= 100:
        return "Moderate", "#facc15"
    if value <= 150:
        return "Unhealthy for Sensitive", "#fb923c"
    if value <= 200:
        return "Unhealthy", "#f87171"
    if value <= 300:
        return "Very Unhealthy", "#c084fc"
    return "Hazardous", "#ef4444"


def pollutant_subindex(value, breakpoints):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= value <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
    c_low, c_high, i_low, i_high = breakpoints[-1]
    return ((i_high - i_low) / (c_high - c_low)) * (min(value, c_high) - c_low) + i_low


def calculate_numeric_aqi(components):
    """Approximate US EPA-style AQI from OpenWeather pollutant concentrations."""
    breakpoints = {
        "pm2_5": [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                  (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
        "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
                 (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
        "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
                (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
        "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
               (86, 105, 151, 200), (106, 200, 201, 300)],
    }

    indexes = [
        pollutant_subindex(components.get(name), bp)
        for name, bp in breakpoints.items()
    ]
    indexes = [idx for idx in indexes if idx is not None]
    return round(max(indexes), 1) if indexes else None


def parse_pollution_item(item):
    components = item.get("components", {})
    dt = datetime.fromtimestamp(item.get("dt", datetime.now().timestamp()))
    return {
        "datetime": dt,
        "aqi": calculate_numeric_aqi(components),
        "openweather_class": item.get("main", {}).get("aqi"),
        **{name: components.get(name) for name in POLLUTANTS},
    }


def openweather_get(endpoint, params):
    url = f"https://api.openweathermap.org/data/2.5/{endpoint}"
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


@st.cache_resource(show_spinner="Loading model artifacts...")
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


@st.cache_data(ttl=1800, show_spinner="Fetching live air quality...")
def fetch_live():
    if not OPENWEATHER_API_KEY:
        return None

    params = {"lat": LATITUDE, "lon": LONGITUDE, "appid": OPENWEATHER_API_KEY}
    weather_params = {**params, "units": "metric"}

    try:
        pollution = openweather_get("air_pollution", params)
        weather = openweather_get("weather", weather_params)
    except requests.RequestException:
        return None

    records = pollution.get("list") or []
    if not records or weather.get("main") is None:
        return None

    live = parse_pollution_item(records[0])
    main = weather.get("main", {})
    wind = weather.get("wind", {})
    live.update({
        "temperature": main.get("temp"),
        "humidity": main.get("humidity"),
        "wind_speed": wind.get("speed"),
        "pressure": main.get("pressure"),
        "station": "Karachi",
    })
    return live


@st.cache_data(ttl=1800, show_spinner="Fetching 3-day forecast...")
def fetch_openweather_forecast():
    if not OPENWEATHER_API_KEY:
        return []

    params = {"lat": LATITUDE, "lon": LONGITUDE, "appid": OPENWEATHER_API_KEY}
    try:
        forecast = openweather_get("air_pollution/forecast", params)
    except requests.RequestException:
        return []

    points = [parse_pollution_item(item) for item in forecast.get("list", [])]
    cutoff = datetime.now() + timedelta(hours=72)
    return [point for point in points if point["datetime"] <= cutoff and point["aqi"] is not None]


@st.cache_data(ttl=3600, show_spinner="Loading history...")
def load_history():
    if not MONGODB_URI:
        return pd.DataFrame()
    try:
        client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=8000,
        )
        records = list(
            client[MONGODB_DB]["aqi_data"]
            .find({}, {"_id": 0})
            .sort("datetime", -1)
            .limit(720)
        )
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime")
    except Exception:
        return pd.DataFrame()


def predict_model_fallback(model, le, feat_info, live):
    predictions = []
    if model is None or le is None or feat_info is None or live is None:
        return predictions

    feature_cols = feat_info.get("selected_features") or feat_info.get("feature_cols") or []
    if not feature_cols:
        return predictions

    df_ext = pd.DataFrame([live])
    last_time = datetime.now()

    for h in range(1, 73):
        next_time = last_time + timedelta(hours=h)
        row = {
            "day_of_year": next_time.timetuple().tm_yday,
            "week": next_time.isocalendar()[1],
        }

        for col in POLLUTANTS:
            row[col] = df_ext[col].iloc[-1] if col in df_ext.columns and not pd.isna(df_ext[col].iloc[-1]) else 0

        pm10_val = row.get("pm10", 0)
        pm2_5_val = row.get("pm2_5", 0)
        no2_val = row.get("no2", 0)
        so2_val = row.get("so2", 0)

        row.update({
            "pm10_lag1": pm10_val,
            "pm10_lag3": pm10_val,
            "pm2_5_lag1": pm2_5_val,
            "aqi_lag1": df_ext[TARGET].iloc[-1] if TARGET in df_ext.columns else live.get("openweather_class", 3),
            "co_lag12": row.get("co", 0),
            "pm_ratio": pm2_5_val / (pm10_val + 1e-6),
            "pm2_5_x_no2": pm2_5_val * no2_val,
            "so2_x_pm10": so2_val * pm10_val,
            "aqi_trend3": 0,
            "aqi_trend6": 0,
        })

        x_vec = np.array([row.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
        try:
            pred_enc = model.predict(x_vec)[0]
            pred_class = int(le.inverse_transform([pred_enc])[0])
            confidence = float(model.predict_proba(x_vec)[0].max()) if hasattr(model, "predict_proba") else None
        except Exception:
            break

        class_to_aqi = {1: 35, 2: 75, 3: 125, 4: 175, 5: 225}
        predictions.append({
            "datetime": next_time,
            "aqi": class_to_aqi.get(pred_class, None),
            "openweather_class": pred_class,
            "confidence": confidence,
        })

        next_row = {name: row.get(name, 0) for name in POLLUTANTS}
        next_row[TARGET] = pred_class
        df_ext = pd.concat([df_ext, pd.DataFrame([next_row])], ignore_index=True)

    return predictions


def summarize_daily(points):
    if not points:
        return []

    df = pd.DataFrame(points)
    df["date_only"] = df["datetime"].dt.date
    summaries = []

    for date_value, group in df.groupby("date_only"):
        if date_value <= datetime.now().date():
            continue
        peak_idx = group["aqi"].astype(float).idxmax()
        peak = group.loc[peak_idx]
        summaries.append({
            "date": pd.to_datetime(date_value),
            "aqi": float(peak["aqi"]),
            "peak_time": peak["datetime"].strftime("%H:%M"),
            "mean_aqi": float(group["aqi"].mean()),
        })
        if len(summaries) == 3:
            break

    return summaries


def figure_layout(height=320):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter"),
        height=height,
        margin=dict(l=4, r=4, t=16, b=4),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zeroline=False),
        showlegend=False,
    )


def render_aqi_card(label, value, footer, large=False):
    category, color = aqi_info(value)
    value_text = "N/A" if value is None or pd.isna(value) else f"{value:.0f}"
    cls = "aqi-value" if large else "forecast-value"
    card_cls = "metric-card" if large else "forecast-card"
    return f"""
    <div class="{card_cls}" style="border-color: {color}66;">
        <div>
            <div class="label">{label}</div>
            <div class="{cls}" style="color: {color};">{value_text}</div>
            <div class="category" style="color: {color};">{category}</div>
        </div>
        <div class="fineprint">{footer}</div>
    </div>
    """


model, le, feat_info = load_model()
live = fetch_live()
history = load_history()
forecast_points = fetch_openweather_forecast()
forecast_source = "OpenWeather pollutant forecast"

if not forecast_points and live is not None:
    forecast_points = predict_model_fallback(model, le, feat_info, live)
    forecast_source = "model fallback"

daily_forecast = summarize_daily(forecast_points)

st.markdown(
    """
<div class="topbar">
    <div>
        <div class="eyebrow">Karachi air quality</div>
        <div class="title">AQI monitor and 72-hour outlook</div>
        <div class="subtitle">
            Live pollutant readings, daily peak forecast, and recent air quality movement on one consistent numeric AQI scale.
        </div>
    </div>
    <div class="status-pill">Forecast source: {source}</div>
</div>
""".format(source=forecast_source),
    unsafe_allow_html=True,
)

if live is None:
    st.error("Could not fetch live data from OpenWeather. Check OPENWEATHER_API_KEY and try again.")
    st.stop()

current_aqi = live["aqi"]
current_cat, current_color = aqi_info(current_aqi)

if current_aqi and current_aqi > 150:
    st.markdown(
        f"""
        <div class="alert-box">
            Health alert: current AQI is <strong>{current_aqi:.0f}</strong> ({current_cat}).
            Limit prolonged outdoor activity, especially for sensitive groups.
        </div>
        """,
        unsafe_allow_html=True,
    )

forecast_cols = st.columns(4)
with forecast_cols[0]:
    st.markdown(
        render_aqi_card(
            "Right now",
            current_aqi,
            f"Updated {live['datetime'].strftime('%b %d, %H:%M')}",
            large=True,
        ),
        unsafe_allow_html=True,
    )

for idx, day in enumerate(daily_forecast[:3], start=1):
    with forecast_cols[idx]:
        footer = f"Peak near {day['peak_time']} - daily avg {day['mean_aqi']:.0f}"
        st.markdown(
            render_aqi_card(day["date"].strftime("%A"), day["aqi"], footer),
            unsafe_allow_html=True,
        )

for idx in range(len(daily_forecast) + 1, 4):
    with forecast_cols[idx]:
        st.markdown(
            render_aqi_card("Forecast", None, "Waiting for forecast data"),
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

chart_left, chart_right = st.columns([1.45, 1])

with chart_left:
    st.markdown('<div class="panel"><div class="section-title">Recent AQI trend</div>', unsafe_allow_html=True)
    if not history.empty and "aqi" in history.columns:
        hist = history.tail(168)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["datetime"],
            y=hist["aqi"],
            mode="lines",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.12)",
            hovertemplate="%{x|%b %d %H:%M}<br>AQI %{y:.0f}<extra></extra>",
        ))
        for y, name, color in [(50, "Good", "#22c55e"), (100, "Moderate", "#facc15"), (150, "Unhealthy", "#f87171")]:
            fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=name, annotation_font_color=color)
        fig.update_layout(**figure_layout(318))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data found yet. Run the backfill/training workflow to populate MongoDB.")
    st.markdown("</div>", unsafe_allow_html=True)

with chart_right:
    st.markdown('<div class="panel"><div class="section-title">Next 72 hours</div>', unsafe_allow_html=True)
    if forecast_points:
        forecast_df = pd.DataFrame(forecast_points)
        sampled = forecast_df.iloc[::3].copy()
        colors = [aqi_info(v)[1] for v in sampled["aqi"]]
        fig = go.Figure(go.Bar(
            x=sampled["datetime"],
            y=sampled["aqi"],
            marker_color=colors,
            hovertemplate="%{x|%b %d %H:%M}<br>AQI %{y:.0f}<extra></extra>",
        ))
        fig.update_layout(**figure_layout(318))
        fig.update_yaxes(title="AQI", range=[0, max(180, float(sampled["aqi"].max()) * 1.15)])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Forecast data is unavailable right now.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Current pollutant readings</div>', unsafe_allow_html=True)

pollutants = {
    "PM2.5": (live.get("pm2_5"), "ug/m3", 35),
    "PM10": (live.get("pm10"), "ug/m3", 50),
    "NO2": (live.get("no2"), "ug/m3", 25),
    "O3": (live.get("o3"), "ug/m3", 60),
    "CO": (live.get("co"), "ug/m3", 4000),
    "SO2": (live.get("so2"), "ug/m3", 20),
}

pollutant_cols = st.columns(6)
for col, (name, (value, unit, reference)) in zip(pollutant_cols, pollutants.items()):
    with col:
        if value is None or pd.isna(value):
            st.markdown(
                f"""
                <div class="pollutant-card">
                    <div class="label">{name}</div>
                    <div class="pollutant-value" style="color:#64748b;">N/A</div>
                    <div class="fineprint">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        ratio = min(float(value) / reference, 1.35)
        color = "#22c55e" if ratio < 0.55 else "#facc15" if ratio < 0.95 else "#f87171"
        width = min(ratio, 1) * 100
        st.markdown(
            f"""
            <div class="pollutant-card">
                <div class="label">{name}</div>
                <div class="pollutant-value" style="color:{color};">{value:.1f}</div>
                <div class="fineprint">{unit}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{width:.0f}%; background:{color};"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.sidebar:
    st.markdown("### Weather")
    weather = {
        "Temperature": f"{live['temperature']} C" if live.get("temperature") is not None else "N/A",
        "Humidity": f"{live['humidity']} %" if live.get("humidity") is not None else "N/A",
        "Wind": f"{live['wind_speed']} m/s" if live.get("wind_speed") is not None else "N/A",
        "Pressure": f"{live['pressure']} hPa" if live.get("pressure") is not None else "N/A",
    }
    for label, value in weather.items():
        st.markdown(f"**{label}:** {value}")

    st.divider()
    st.markdown("### Data")
    st.caption(f"Station: {live.get('station', 'Karachi')}")
    st.caption(f"Current AQI category: {current_cat}")
    st.caption(f"Forecast points: {len(forecast_points)}")
    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.divider()
    st.markdown("### Model")
    if model is not None:
        features = (feat_info.get("selected_features") or feat_info.get("feature_cols") or []) if feat_info else []
        st.success("Artifacts loaded")
        st.caption("The trained model is kept as a fallback when the OpenWeather forecast is unavailable.")
        st.caption(f"Features: {len(features)}")
    else:
        st.warning("Model artifacts not found")

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

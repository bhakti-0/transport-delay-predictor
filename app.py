import streamlit as st
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from src.predict import predict_with_data

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Transport Delay Predictor", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🚍 Public Transport Delay Predictor
</h1>
<p style='text-align: center;'>
Predict delays using weather & operational factors
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("📅 Select Date")
    selected_time = st.time_input("⏰ Select Time")

    temperature = st.number_input("🌡 Temperature (°C)", value=25.0)
    humidity = st.number_input("💧 Humidity (%)", value=60.0)
    wind = st.number_input("🌬 Wind Speed (km/h)", value=10.0)

with col2:
    rain = st.selectbox("🌧 Rain (0 = No, 1 = Yes)", [0, 1])
    event = st.selectbox("🎉 Event (0 = No, 1 = Yes)", [0, 1])
    traffic = st.slider("🚦 Traffic Congestion Index", 0, 10, 5)

# Show selected datetime nicely
selected_datetime = datetime.combine(selected_date, selected_time)
st.info(f"🕒 Selected: {selected_datetime}")

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Delay"):

    input_data = {
        "date": str(selected_date),
        "time": str(selected_time),
        "temperature_C": temperature,
        "humidity_percent": humidity,
        "wind_speed_kmh": wind,
        "precipitation_mm": rain,
        "event_type": event,
        "traffic_congestion_index": traffic
    }

    result, df = predict_with_data(input_data)

    if result is None:
        st.error("⚠️ Model not trained. Please run training first.")
    else:
        st.success(f"⏱ Estimated Delay: **{round(result, 2)} minutes**")

        # ---- Delay Category ----
        if result < 5:
            st.info("🟢 Low delay expected")
        elif result < 15:
            st.warning("🟡 Moderate delay")
        else:
            st.error("🔴 High delay expected")

        # ---------------- SHAP ----------------
        st.subheader("🧠 Prediction Explanation (SHAP)")

        try:
            import joblib
            model = joblib.load("model.pkl")

            explainer = shap.Explainer(model)
            shap_values = explainer(df)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"SHAP visualization error: {e}")
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go

# --- 1. Page Configuration & Theme Settings ---
st.set_page_config(
    page_title="OrbitTrak AI Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling injection for an engineering dashboard look
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #ffffff; }
    .stSlider { padding-top: 10px; padding-bottom: 10px; }
    h1 { color: #00e5ff !important; font-family: 'Courier New', monospace; }
    h2, h3 { color: #e0e0e0 !important; }
    div[data-testid="stMetricValue"] { color: #00ff66 !important; font-family: 'Share Tech Mono', monospace; }
    </style>
""", unsafe_html=True)

st.title("OrbitTrak AI: Neural Flight Computer & Trajectory Predictor")
st.markdown("---")

# --- 2. Safe Asset Ingestion Engine ---
@st.cache_resource
def load_flight_assets():
    """Reads the pre-compiled TensorFlow model and scaler transformation coefficients."""
    nn_model = tf.keras.models.load_model('orbit_predictor_model.h5')
    with open('scaler.pkl', 'rb') as f:
        data_scaler = pickle.load(f)
    return nn_model, data_scaler

try:
    model, scaler = load_flight_assets()
except Exception as e:
    st.error("System Core Error: Critical Assets Missing. Run data_gen.py and train_model.py first.")
    st.info(f"Log Output: {e}")
    st.stop()

# --- 3. Dashboard Interface Architecture ---
col_inputs, col_viz = st.columns([1, 1.3], gap="large")

with col_inputs:
    st.header("Flight Control Parameters")
    st.caption("Adjust engineering variables. The neural network computes telemetry modifications immediately without running numerical integration loops.")
    
    # User Inputs via Sliders
    angle = st.slider("Launch Pitch Vector Angle (°)", min_value=10.0, max_value=90.0, value=45.0, step=0.5)
    duration = st.slider("Main Booster Burn Window Duration (s)", min_value=20.0, max_value=180.0, value=90.0, step=1.0)
    power = st.slider("Engine Core Thrust Execution Power (kN)", min_value=200.0, max_value=1200.0, value=600.0, step=10.0)
    mass = st.slider("Total Spacecraft Wet Structural Mass (kg)", min_value=800.0, max_value=4000.0, value=2000.0, step=50.0)
    
    st.markdown("---")
    st.header("AI Live Diagnostic Readout")
    
    # --- 4. Live Model Inference Processing ---
    input_vector = np.array([[angle, duration, power, mass]])
    scaled_vector = scaler.transform(input_vector)
    
    # Run predictions across regression and classification branches
    predictions = model.predict(scaled_vector, verbose=0)
    predicted_heights = predictions[0][0]  
    predicted_stability = predictions[1][0][0]  
    
    # Separate prediction values and account for boundary limits
    pred_apo = max(0.0, predicted_heights[0])
    pred_peri = max(0.0, predicted_heights[1])
    
    # Render Metric Readouts
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Predicted Apoapsis Altitude", value=f"{pred_apo / 1000:.2f} km")
    with metric_col2:
        st.metric(label="Predicted Periapsis Altitude", value=f"{pred_peri / 1000:.2f} km")
        
    # Evaluate Sigmoid Confidence Thresholds
    if predicted_stability >= 0.5:
        st.success(f"Stable Orbit Confirmed (Model Confidence: {predicted_stability * 100:.1f}%)")
    else:
        st.error(f"Trajectory Alert: Surface Collision / Atmosphere Intercept (Stability Metric: {predicted_stability * 100:.1f}%)")

with col_viz:
    st.header("Interactive Flight Path Projection")
    
    # Baseline Earth Radius Constant Metrics (in kilometers)
    R_EARTH_KM = 6371.0
    R_ATMOSPHERE_KM = R_EARTH_KM + 100.0
    
    # Compute circular array map representing Earth and Atmosphere
    theta = np.linspace(0, 2 * np.pi, 200)
    earth_x = R_EARTH_KM * np.cos(theta)
    earth_y = R_EARTH_KM * np.sin(theta)
    
    atmo_x = R_ATMOSPHERE_KM * np.cos(theta)
    atmo_y = R_ATMOSPHERE_KM * np.sin(theta)
    
    # Map out the elliptical trajectory projection paths based on model telemetry heights
    apo_radius_km = (pred_apo / 1000.0) + R_EARTH_KM
    peri_radius_km = (pred_peri / 1000.0) + R_EARTH_KM
    
    # Coordinate generation mapping focal center displacements
    orbit_x = apo_radius_km * np.cos(theta)
    orbit_y = peri_radius_km * np.sin(theta)
    
    # Initialize Plotly Canvas
    fig = go.Figure()
    
    # Trace 1: Draw the Planetary Target Core
    fig.add_trace(go.Scatter(
        x=earth_x, y=earth_y,
        fill="toself",
        fillcolor="rgba(0, 150, 255, 0.2)",
        line=dict(color="cyan", width=2),
        name="Earth Surface Core"
    ))
    
    # Trace 2: Draw the Atmospheric Boundary Layer (Kármán Line)
    fig.add_trace(go.Scatter(
        x=atmo_x, y=atmo_y,
        mode="lines",
        line=dict(color="rgba(255, 0, 0, 0.35)", width=1.5, dash="dot"),
        name="Atmosphere Limit (100 km)"
    ))
    
    # Trace 3: Draw the real-time AI Forecast Path Line
    fig.add_trace(go.Scatter(
        x=orbit_x, y=orbit_y,
        mode="lines",
        line=dict(color="#ff9100", width=3, dash="dashdot"),
        name="Neural Network Vector Path"
    ))
    
    # Layout and Viewport Formatting Rules
    fig.update_layout(
        xaxis=dict(title="Radial X Distance Axis (km)", range=[-18000, 18000], gridcolor="#1e293b", scaleratio=1, scaleanchor="y"),
        yaxis=dict(title="Radial Y Distance Axis (km)", range=[-18000, 18000], gridcolor="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=700,
        height=700,
        showlegend=True,
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(15, 23, 42, 0.8)")
    )
    
    st.plotly_chart(fig, use_container_width=True)

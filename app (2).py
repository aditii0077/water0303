
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sqlite3
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# Database setup
conn = sqlite3.connect("water_quality.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS assessments
                  (timestamp TEXT, ph REAL, hardness REAL, solids REAL, chloramines REAL,
                   sulfate REAL, conductivity REAL, organic_carbon REAL,
                   trihalomethanes REAL, turbidity REAL, result TEXT)''')
conn.commit()

# Load dataset
df = pd.read_csv("water_potability.csv")
df.fillna(df.mean(), inplace=True)
X = df.drop("Potability", axis=1)
y = df["Potability"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train AI models
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

dl_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Streamlit UI
st.set_page_config(page_title="Water Quality Assessment", layout="wide")
st.title("💧 Water Quality Assessment Tool")

# User Inputs
st.sidebar.header("🔍 Enter Water Quality Parameters")
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness", 0.0, 500.0, 150.0)
solids = st.sidebar.slider("Solids (ppm)", 0.0, 50000.0, 20000.0)
chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 250.0)
conductivity = st.sidebar.slider("Conductivity", 0.0, 1000.0, 500.0)
organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 50.0, 20.0)
trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 150.0, 75.0)
turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 4.0)

# AI Prediction
input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
input_scaled = scaler.transform(input_data)
rf_pred = rf_model.predict(input_scaled)[0]
dl_pred = dl_model.predict(input_scaled)[0][0]

result = "Potable" if rf_pred == 1 or dl_pred > 0.5 else "Not Potable"
st.subheader("🔬 AI Analysis")
if result == "Potable":
    st.success("✅ The water is likely **Potable (Safe to Drink)**")
else:
    st.error("🚨 The water is **Not Potable (Unsafe to Drink)**")

# Store result in database
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cursor.execute("INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (timestamp, ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity, result))
conn.commit()

# Trend Analysis
st.subheader("📊 Trend Analysis")
fig = px.histogram(df, x="ph", color="Potability", nbins=50, title="pH Distribution")
st.plotly_chart(fig)

# Time-based Water Safety Analysis
st.subheader("⏳ Time-Based Water Safety Analysis")
data = pd.read_sql("SELECT * FROM assessments", conn)
if not data.empty:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    fig = px.line(data, x='timestamp', y=['ph', 'turbidity', 'organic_carbon'], title="Water Quality Changes Over Time")
    st.plotly_chart(fig)

# Water Source Comparison
st.subheader("🔄 Water Quality Source Comparison")
source_options = ["Groundwater", "River", "Tap Water", "Bottled Water"]
source_choice = st.selectbox("Select Water Source", source_options)

if source_choice == "Groundwater":
    st.info("Groundwater is generally hard and may contain high levels of dissolved minerals.")
elif source_choice == "River":
    st.info("River water can be affected by industrial pollution and bacteria.")
elif source_choice == "Tap Water":
    st.info("Tap water is usually treated but may contain chlorine and contaminants.")
else:
    st.info("Bottled water is purified but may have plastic micro-particles.")

# Upload Image for Visual Analysis
st.subheader("📸 Upload Water Sample Image for Visual Analysis")
uploaded_file = st.file_uploader("Upload an image of the water sample", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(edges, caption="Impurity Detection", use_column_width=True)

# Download Reports
st.subheader("📂 Download Report")
st.download_button("Download Data Report", data.to_csv().encode(), "water_quality_report.csv", "text/csv")

st.markdown("---")
st.markdown("🔗 Developed by **Aditi Kalbhor** | Powered by AI & ML 🚀")

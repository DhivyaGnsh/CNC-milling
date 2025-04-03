import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# Set Page Configuration
st.set_page_config(page_title="Tool Wear Prediction", page_icon="🔧", layout="wide")

# File Paths
MODEL_PATH = "E:/StreamLit/env/Scripts/MODEL (2).h5"
SCALER_PATH = "E:/StreamLit/env/Scripts/scaler (6).pkl"

# Load Model and Scaler with Error Handling
try:
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

# Define Feature Columns
selected_columns = [
    "material", "feedrate", "clamp_pressure", "X1_ActualPosition", "Y1_ActualPosition", "Z1_ActualPosition",
    "X1_CurrentFeedback", "Y1_CurrentFeedback", "M1_CURRENT_FEEDRATE", "X1_DCBusVoltage",
    "X1_OutputPower", "Y1_OutputPower", "S1_OutputPower"
]

sequence_length = 10  # LSTM expects sequential data

# Sidebar Styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #003366, #0066cc, #00bfff);
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
menu = ["🏠 Home", "📊 Prediction"]
choice = st.sidebar.radio("Navigation", menu)

# Home Page
if choice == "🏠 Home":
    st.title("🔬 Tool Wear Prediction using LSTM")
    st.markdown("""
        ## 🧠 About the Model
        - **Model Type:** LSTM (Long Short-Term Memory)
        - **Purpose:** Predict tool wear conditions based on machining parameters
        - **How It Works?**
          - Enter real-time parameters.
          - The model predicts tool wear and inspection results.

        ## 🌍 Why is This Useful?
        ✅ Reduces **downtime** by predicting tool wear in advance  
        ✅ Saves **costs** by avoiding unnecessary tool replacements  
        ✅ Ensures **better machining quality** with AI-driven insights  
    """)

# Prediction Page
elif choice == "📊 Prediction":
    st.title("🔍 Tool Wear Prediction")

    # Sidebar Inputs
    st.sidebar.header("📥 Enter Feature Values")
    user_input = {feature: st.sidebar.number_input(f"{feature}", value=0.0) for feature in selected_columns}

    # Preprocess Input
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    input_sequence = np.array([input_scaled] * sequence_length).reshape(1, sequence_length, len(selected_columns))

    # Predict Button
    if st.sidebar.button("🚀 Predict"):
        try:
            pred_tool_condition, pred_machining_finalized, pred_visual_inspection = model.predict(input_sequence)

            # Convert Predictions to Labels
            tool_condition_labels = {0: "🟢 Good", 1: "🟠 Worn", 2: "🔴 Damaged"}
            pred_tool_condition_label = tool_condition_labels[np.argmax(pred_tool_condition[0])]
            machining_finalized_status = "✅ Yes" if pred_machining_finalized[0][0] > 0.5 else "❌ No"
            visual_inspection_status = "✔️ Passed" if pred_visual_inspection[0][0] > 0.5 else "❌ Failed"

            # Display Results
            st.subheader("📈 Prediction Results")
            st.write(f"**🛠 Tool Condition:** {pred_tool_condition_label}")
            st.write(f"**🔄 Machining Finalized:** {machining_finalized_status}")
            st.write(f"**👀 Visual Inspection:** {visual_inspection_status}")
            st.success("✅ Prediction Completed!")

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

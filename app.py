import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import random

# ---------------- LOAD MODEL ----------------
model = joblib.load("electricity_theft_model.pkl")

st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("AI-based smart electricity theft risk detection")

# ---------------- SAMPLE GENERATOR ----------------
def generate_sample():
    return {
        "usage_1": random.randint(50, 200),
        "usage_2": random.randint(0, 150),
        "usage_3": random.randint(0, 150),
        "usage_4": random.randint(0, 150),

        "mtr_val_old": random.randint(3000, 6000),
        "mtr_val_new": random.randint(3000, 6500),

        "mtr_coef": round(random.uniform(0.8, 1.5), 2),
        "months_num": random.randint(1, 12),

        "mtr_tariff": 10,
        "mtr_status": 0,
        "mtr_code": 200,
        "mtr_notes": 5,
        "mtr_type": 0,
        "usage_aux": 0,
        "usage_n_aux": 0,
        "date_flip_flag": 0,
        "date_overlap_invoice": 0,
        "date_overlap_months": 0,
        "months_num_calc": 5.0
    }

if "data" not in st.session_state:
    st.session_state.data = generate_sample()

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Generate Sample"):
        st.session_state.data = generate_sample()

with col2:
    if st.button("🔄 Reset"):
        st.session_state.data = generate_sample()

data = st.session_state.data

# ---------------- INPUT UI ----------------
st.subheader("📥 Enter Key Details")

usage_1 = st.number_input("usage_1", value=float(data["usage_1"]))
usage_2 = st.number_input("usage_2", value=float(data["usage_2"]))
usage_3 = st.number_input("usage_3", value=float(data["usage_3"]))
usage_4 = st.number_input("usage_4", value=float(data["usage_4"]))

mtr_val_old = st.number_input("mtr_val_old", value=float(data["mtr_val_old"]))
mtr_val_new = st.number_input("mtr_val_new", value=float(data["mtr_val_new"]))

mtr_coef = st.number_input("mtr_coef", value=float(data["mtr_coef"]))
months_num = st.number_input("months_num", value=float(data["months_num"]))

# ---------------- BUILD FULL INPUT ----------------
full_input = {
    "mtr_tariff": 10,
    "mtr_status": 0,
    "mtr_code": 200,
    "mtr_notes": 5,

    "mtr_coef": mtr_coef,

    "usage_1": usage_1,
    "usage_2": usage_2,
    "usage_3": usage_3,
    "usage_4": usage_4,

    "mtr_val_old": mtr_val_old,
    "mtr_val_new": mtr_val_new,
    "months_num": months_num,

    "mtr_type": 0,
    "usage_aux": 0,
    "usage_n_aux": usage_1 + usage_2 + usage_3 + usage_4,

    "date_flip_flag": 0,
    "date_overlap_invoice": 0,
    "date_overlap_months": 0,
    "months_num_calc": months_num * 1.1
}

# ---------------- FIX: SAFE COLUMN ORDERING ----------------
if hasattr(model, "feature_names_in_"):
    feature_order = list(model.feature_names_in_)
else:
    feature_order = list(full_input.keys())

input_df = pd.DataFrame([full_input])[feature_order]

# ---------------- PREDICTION ----------------
def get_probs(model, data):
    probs = model.predict_proba(data)[0]
    return probs[0], probs[1]

def risk_level(p):
    if p >= 0.80:
        return "🔴 HIGH RISK"
    elif p >= 0.50:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def plot_graph(p0, p1):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Theft Risk", "Normal Usage"],
        y=[p0, p1],
        text=[f"{p0:.2f}", f"{p1:.2f}"],
        textposition="auto",
        marker_color=["red", "green"]
    ))
    fig.update_layout(title="⚡ Risk Probability", height=400)
    return fig

# ---------------- FEATURE IMPORTANCE ----------------
def show_feature_importance():
    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame({
            "Feature": feature_order,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        return df
    else:
        return pd.DataFrame({"Message": ["Feature importance not available for this model"]})

# ---------------- ANALYZE ----------------
if st.button("🔍 Analyze"):

    p_theft, p_normal = get_probs(model, input_df)

    st.subheader("📊 Result")

    st.write(f"⚠️ Theft Probability: {p_theft:.2f}")
    st.write(f"✅ Normal Probability: {p_normal:.2f}")

    if p_theft >= 0.80:
        st.error(risk_level(p_theft))
    elif p_theft >= 0.50:
        st.warning(risk_level(p_theft))
    else:
        st.success(risk_level(p_theft))

    st.subheader("📈 Visualization")
    st.plotly_chart(plot_graph(p_theft, p_normal), use_container_width=True)

    st.subheader("🔍 Feature Importance (Model Insight)")
    st.dataframe(show_feature_importance())

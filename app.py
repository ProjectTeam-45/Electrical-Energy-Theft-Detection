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
st.write("AI-based electricity theft risk analysis system")

# ---------------- FEATURE ORDER ----------------
features = [
    'mtr_tariff', 'mtr_status', 'mtr_code', 'mtr_notes', 'mtr_coef',
    'usage_1', 'usage_2', 'usage_3', 'usage_4',
    'mtr_val_old', 'mtr_val_new', 'months_num',
    'mtr_type', 'usage_aux', 'usage_n_aux',
    'date_flip_flag', 'date_overlap_invoice', 'date_overlap_months',
    'months_num_calc'
]

# ---------------- AUTO SAMPLE GENERATOR ----------------
def generate_sample():
    return {
        "mtr_tariff": random.randint(5, 15),
        "mtr_status": random.randint(0, 1),
        "mtr_code": random.randint(100, 300),
        "mtr_notes": random.randint(1, 10),
        "mtr_coef": round(random.uniform(0.8, 1.5), 2),

        "usage_1": random.randint(50, 200),
        "usage_2": random.randint(0, 150),
        "usage_3": random.randint(0, 150),
        "usage_4": random.randint(0, 150),

        "mtr_val_old": random.randint(3000, 6000),
        "mtr_val_new": random.randint(3000, 6500),
        "months_num": random.randint(1, 12),

        "mtr_type": random.randint(0, 1),
        "usage_aux": random.randint(0, 200),
        "usage_n_aux": random.randint(0, 200),

        "date_flip_flag": random.randint(0, 1),
        "date_overlap_invoice": random.randint(0, 1),
        "date_overlap_months": random.randint(0, 1),

        "months_num_calc": round(random.uniform(1, 12), 1)
    }

# ---------------- SESSION STATE ----------------
if "data" not in st.session_state:
    st.session_state.data = generate_sample()

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Generate Sample Data"):
        st.session_state.data = generate_sample()

with col2:
    if st.button("🔄 Reset"):
        st.session_state.data = generate_sample()

data = st.session_state.data

# ---------------- MANUAL INPUT (NO LOOP - YOUR STYLE) ----------------
st.subheader("📥 Enter Meter Details")

mtr_tariff = st.number_input("mtr_tariff", value=float(data["mtr_tariff"]))
mtr_status = st.number_input("mtr_status", value=float(data["mtr_status"]))
mtr_code = st.number_input("mtr_code", value=float(data["mtr_code"]))
mtr_notes = st.number_input("mtr_notes", value=float(data["mtr_notes"]))
mtr_coef = st.number_input("mtr_coef", value=float(data["mtr_coef"]))

usage_1 = st.number_input("usage_1", value=float(data["usage_1"]))
usage_2 = st.number_input("usage_2", value=float(data["usage_2"]))
usage_3 = st.number_input("usage_3", value=float(data["usage_3"]))
usage_4 = st.number_input("usage_4", value=float(data["usage_4"]))

mtr_val_old = st.number_input("mtr_val_old", value=float(data["mtr_val_old"]))
mtr_val_new = st.number_input("mtr_val_new", value=float(data["mtr_val_new"]))
months_num = st.number_input("months_num", value=float(data["months_num"]))

mtr_type = st.number_input("mtr_type", value=float(data["mtr_type"]))
usage_aux = st.number_input("usage_aux", value=float(data["usage_aux"]))
usage_n_aux = st.number_input("usage_n_aux", value=float(data["usage_n_aux"]))

date_flip_flag = st.number_input("date_flip_flag", value=float(data["date_flip_flag"]))
date_overlap_invoice = st.number_input("date_overlap_invoice", value=float(data["date_overlap_invoice"]))
date_overlap_months = st.number_input("date_overlap_months", value=float(data["date_overlap_months"]))

months_num_calc = st.number_input("months_num_calc", value=float(data["months_num_calc"]))

# ---------------- INPUT DF ----------------
input_df = pd.DataFrame([[
    mtr_tariff, mtr_status, mtr_code, mtr_notes, mtr_coef,
    usage_1, usage_2, usage_3, usage_4,
    mtr_val_old, mtr_val_new, months_num,
    mtr_type, usage_aux, usage_n_aux,
    date_flip_flag, date_overlap_invoice, date_overlap_months,
    months_num_calc
]], columns=features)

# ---------------- LOGIC ----------------
def get_probs(model, data):
    probs = model.predict_proba(data)[0]
    return probs[0], probs[1]   # 0 = Theft, 1 = Normal

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

    fig.update_layout(
        title="⚡ Electricity Theft Probability",
        height=400
    )

    return fig

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

    st.subheader("📋 Input Data")
    st.dataframe(input_df)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------- LOAD MODEL ----------------
model = joblib.load("electricity_theft_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("Enter meter details to analyze theft risk")

# ---------------- FEATURE ORDER (VERY IMPORTANT) ----------------
features = [
    'mtr_tariff', 'mtr_status', 'mtr_code', 'mtr_notes', 'mtr_coef',
    'usage_1', 'usage_2', 'usage_3', 'usage_4',
    'mtr_val_old', 'mtr_val_new', 'months_num',
    'mtr_type', 'usage_aux', 'usage_n_aux',
    'date_flip_flag', 'date_overlap_invoice', 'date_overlap_months',
    'months_num_calc'
]

# ---------------- INPUT UI ----------------
st.subheader("📥 Enter Meter Details")

user_input = {}
for col in features:
    user_input[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([user_input])
input_df = input_df[features]

# ---------------- PROBABILITY FUNCTION ----------------
def get_probs(model, data):
    probs = model.predict_proba(data)[0]

    prob_theft = probs[0]   # 0 = Theft
    prob_normal = probs[1]  # 1 = Normal

    return prob_theft, prob_normal

# ---------------- RISK LEVEL ----------------
def get_risk(prob_theft):
    if prob_theft >= 0.80:
        return "🔴 HIGH RISK"
    elif prob_theft >= 0.50:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# ---------------- GRAPH ----------------
def plot_probabilities(prob_theft, prob_normal):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Theft Risk", "Normal Usage"],
        y=[prob_theft, prob_normal],
        text=[f"{prob_theft:.2f}", f"{prob_normal:.2f}"],
        textposition="auto",
        marker_color=["red", "green"]
    ))

    fig.update_layout(
        title="⚡ Theft Risk Probability",
        yaxis_title="Probability",
        height=400
    )

    return fig

# ---------------- FEATURE IMPORTANCE ----------------
def explain_prediction():
    df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return df.head(5)

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze"):

    prob_theft, prob_normal = get_probs(model, input_df)
    risk = get_risk(prob_theft)

    st.subheader("📊 Results")

    if prob_theft >= 0.80:
        st.error(f"{risk} ({prob_theft:.2f})")
    elif prob_theft >= 0.50:
        st.warning(f"{risk} ({prob_theft:.2f})")
    else:
        st.success(f"{risk} ({prob_theft:.2f})")

    st.write(f"⚠️ Theft Probability: {prob_theft:.2f}")
    st.write(f"✅ Normal Probability: {prob_normal:.2f}")

    st.subheader("📈 Probability Visualization")
    st.plotly_chart(plot_probabilities(prob_theft, prob_normal), use_container_width=True)

    st.subheader("🔍 Top Influencing Features")
    st.dataframe(explain_prediction())

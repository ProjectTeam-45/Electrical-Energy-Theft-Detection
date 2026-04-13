import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgb_model.pkl")
imputer = joblib.load("imputer.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("AI-based system to detect electricity theft risk.")

# ---------------- FIXED FEATURES ----------------
features = [
    "usage_1",
    "usage_2",
    "usage_3",
    "usage_4",
    "mtr_val_old",
    "mtr_val_new",
    "months_num",
    "mtr_status"
]

# ---------------- INPUT UI ----------------
st.subheader("📥 Enter Meter Details")

default_values = {
    "usage_1": 100,
    "usage_2": 100,
    "usage_3": 100,
    "usage_4": 100,
    "mtr_val_old": 500,
    "mtr_val_new": 550,
    "months_num": 4,
    "mtr_status": 1
}

user_input = {}

for col in features:
    user_input[col] = st.number_input(col, value=float(default_values[col]))

# ---------------- DATA PREP ----------------
input_df = pd.DataFrame([user_input])

# SAFE: enforce training feature order manually
input_df = input_df[features]

# ---------------- IMPUTATION ----------------
input_df = imputer.transform(input_df)

# ---------------- RISK FUNCTIONS ----------------
def get_risk_level(prob_theft):
    if prob_theft < 0.30:
        return "🟢 Low Risk"
    elif prob_theft < 0.45:
        return "🟡 Medium Risk"
    elif prob_theft < 0.65:
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"


def explain_risk(prob_theft):
    if prob_theft < 0.30:
        return "Normal usage pattern detected."
    elif prob_theft < 0.45:
        return "Minor irregularities in usage pattern."
    elif prob_theft < 0.65:
        return "Strong deviation from normal usage detected."
    else:
        return "Highly suspicious consumption pattern detected."

# ---------------- PROBABILITY ----------------
def get_probs(model, data):
    probs = model.predict_proba(data)[0]

    # FIX: safe mapping (0 = Theft, 1 = Normal)
    return probs[0], probs[1]

# ---------------- GRAPH ----------------
def plot_probs(theft, normal):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Theft", "Normal"],
        y=[theft, normal],
        text=[f"{theft:.2f}", f"{normal:.2f}"],
        textposition="auto"
    ))

    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        height=400
    )

    return fig

# ---------------- FEATURE IMPORTANCE ----------------
def feature_importance():
    try:
        df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        return df.head(5)
    except:
        return pd.DataFrame({"Message": ["Feature importance not available"]})

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Risk"):

    prob_theft, prob_normal = get_probs(model, input_df)

    risk = get_risk_level(prob_theft)
    explanation = explain_risk(prob_theft)

    st.subheader("📊 Results")

    if prob_theft >= 0.65:
        st.error(f"{risk} ({prob_theft:.2f})")
    elif prob_theft >= 0.45:
        st.warning(f"{risk} ({prob_theft:.2f})")
    else:
        st.success(f"{risk} ({prob_theft:.2f})")

    st.write(f"🔴 Theft Probability: {prob_theft:.2f}")
    st.write(f"🟢 Normal Probability: {prob_normal:.2f}")

    st.subheader("📈 Probability Chart")
    st.plotly_chart(plot_probs(prob_theft, prob_normal), use_container_width=True)

    st.subheader("🧠 Explanation")
    st.info(explanation)

    st.subheader("🔍 Feature Importance")
    st.dataframe(feature_importance())

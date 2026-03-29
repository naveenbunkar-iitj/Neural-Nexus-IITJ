import requests
import streamlit as st

st.set_page_config(page_title="Smart Decision Engine", page_icon="📊", layout="wide")

st.title("📊 Smart Decision Engine for Risk & Investment Approval")
st.caption("Hybrid model: Decision Tree + Bayesian reasoning + scoring rules")

with st.sidebar:
    st.header("Applicant Inputs")
    income = st.number_input("Annual Income ($)", min_value=0.0, value=65000.0, step=1000.0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=690)
    liabilities = st.number_input("Total Liabilities ($)", min_value=0.0, value=22000.0, step=500.0)
    expense_ratio = st.slider("Expense Ratio", min_value=0.0, max_value=1.0, value=0.45)
    behavior_score = st.slider("Behavior Score", min_value=0.0, max_value=1.0, value=0.65)

payload = {
    "income": income,
    "credit_score": float(credit_score),
    "liabilities": liabilities,
    "expense_ratio": expense_ratio,
    "behavior_score": behavior_score,
}

if st.button("Run Decision", type="primary"):
    try:
        response = requests.post("http://localhost:8000/decision", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", data["decision"])
        c2.metric("Confidence", f"{data['confidence']}%")
        c3.metric("Risk Score", f"{data['risk_score']}%")

        st.subheader("Explainable AI Output")
        for item in data["explanation"]:
            st.write(f"- {item}")

        st.subheader("Model Internals")
        st.json(data["model_outputs"])

    except Exception as exc:
        st.error(f"API error: {exc}")
        st.info("Start backend first: uvicorn backend.main:app --reload")
else:
    st.info("Configure inputs and click **Run Decision**.")

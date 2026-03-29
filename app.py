import streamlit as st

from backend.decision_engine import SmartDecisionEngine

st.set_page_config(page_title="AI Decision Engine", page_icon="🤖", layout="centered")
st.title("🤖 AI Decision Engine")
st.caption("Hackathon-friendly Streamlit app with decision, risk score, and explanation.")

engine = SmartDecisionEngine()

income = st.number_input("Income", min_value=0.0, value=65000.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=690.0, step=1.0)
liabilities = st.number_input("Debt / Liabilities", min_value=0.0, value=22000.0, step=500.0)
expense_ratio = st.slider("Expense Ratio", min_value=0.0, max_value=1.0, value=0.45)
behavior_score = st.slider("Behavior Score", min_value=0.0, max_value=1.0, value=0.65)

if st.button("Predict", type="primary"):
    payload = {
        "income": income,
        "credit_score": credit_score,
        "liabilities": liabilities,
        "expense_ratio": expense_ratio,
        "behavior_score": behavior_score,
    }
    result = engine.evaluate(payload)

    c1, c2 = st.columns(2)
    c1.metric("Decision", result.decision)
    c2.metric("Risk Score", f"{result.risk_score}%")
    st.metric("Confidence", f"{result.confidence}%")

    st.subheader("Explanation")
    for reason in result.explanation:
        st.write(f"- {reason}")

    st.subheader("Model Outputs")
    st.json(result.model_outputs)
else:
    st.info("Enter applicant details and click Predict.")

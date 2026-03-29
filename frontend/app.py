import requests
import streamlit as st

st.title("🧠 AI Decision Engine")

income = st.slider("Income", 0, 100000, 50000)
credit = st.slider("Credit Score", 300, 900, 650)
debt = st.slider("Debt", 0, 50000, 10000)

if st.button("Evaluate"):
    url = "https://YOUR-API.onrender.com/predict"  # change after deploy

    response = requests.post(
        url,
        params={
            "income": income,
            "credit": credit,
            "debt": debt,
        },
        timeout=15,
    )

    result = response.json()

    st.success(f"Decision: {result['decision']}")
    st.metric("Risk Score", result["risk_score"])

import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Create dummy training data
X = [
    [50000, 700, 10000],
    [30000, 500, 40000],
    [80000, 750, 5000],
    [20000, 400, 30000]
]

y = ["APPROVED", "REJECTED", "APPROVED", "REJECTED"]

# Train model inside app
model = DecisionTreeClassifier()
model.fit(X, y)

st.title("🧠 AI Decision Engine")

income = st.slider("Income", 0, 100000, 50000)
credit = st.slider("Credit Score", 300, 900, 650)
debt = st.slider("Debt", 0, 50000, 10000)

if st.button("Evaluate"):
    result = model.predict([[income, credit, debt]])
    st.success(f"Decision: {result[0]}")

    st.subheader("Explanation")
    if credit < 600:
        st.warning("Low credit score → High risk")
    if debt > 30000:
        st.warning("High debt → Risk factor")

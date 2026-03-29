
import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model_file.pkl', 'rb'))

st.set_page_config(page_title="AI Decision Engine")

st.title("🧠 AI Decision Engine")
st.write("Automation of Complex Decision Making")

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

"""Standalone Streamlit deployment entrypoint.

This file intentionally loads the persisted .pkl model directly so platforms
that only run `streamlit run app.py` work without requiring the FastAPI server.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from backend.decision_engine import SmartDecisionEngine

MODEL_PATH = Path("models/decision_tree.pkl")


def load_model():
    """Load a persisted model, training once if the artifact does not exist yet."""
    if not MODEL_PATH.exists():
        SmartDecisionEngine(model_path=MODEL_PATH)

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def main() -> None:
    st.set_page_config(page_title="AI Decision Engine", page_icon="🤖")
    st.title("AI Decision Engine")
    st.caption("Deployable Streamlit UI with direct .pkl model loading")

    model = load_model()

    income = st.number_input("Income", min_value=0.0, value=65000.0, step=1000.0)
    credit = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=700.0, step=1.0)
    debt = st.number_input("Debt", min_value=0.0, value=20000.0, step=500.0)

    # Decision tree expects [income, credit_score, liabilities, expense_ratio, behavior_score]
    features = np.array([[income, credit, debt, 0.45, 0.65]], dtype=float)

    if st.button("Predict", type="primary"):
        prediction = model.predict(features)[0]
        label = "APPROVE" if int(prediction) == 1 else "REJECT"
        st.success(f"Decision: {label}")


if __name__ == "__main__":
    main()

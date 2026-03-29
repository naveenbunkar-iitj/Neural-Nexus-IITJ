# Smart Decision Engine for Financial Risk & Investment Approval

A hackathon-ready hybrid AI system that automates complex financial decisions with explainable output.

## Features

- **Loan / investment decisioning**: `APPROVE`, `REJECT`, or `HOLD`
- **Risk scoring**: weighted score from user profile and behavior
- **Hybrid inference**:
  - Decision Tree (ML)
  - Bayesian-style probabilistic confidence
  - Rule-based aggregation
- **Explainable AI output** with judge-friendly reason strings
- **FastAPI backend + Streamlit dashboard**

## Project Structure

```text
AI-Decision-Engine/
├── data/
├── models/
├── backend/
│   ├── main.py
│   └── decision_engine.py
├── frontend/
│   └── app.py
├── utils/
│   └── sample_payloads.json
├── notebooks/
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Run backend

```bash
uvicorn backend.main:app --reload
```

API docs: <http://localhost:8000/docs>

### 2) Run frontend

```bash
streamlit run frontend/app.py
```

## API Example

```bash
curl -X POST http://localhost:8000/decision \
  -H "Content-Type: application/json" \
  -d '{
    "income": 75000,
    "credit_score": 710,
    "liabilities": 25000,
    "expense_ratio": 0.48,
    "behavior_score": 0.62
  }'
```

## Decision Logic (Summary)

1. Train a synthetic-data Decision Tree classifier.
2. Compute normalized risk score from:
   - credit profile
   - debt-to-income
   - expense ratio
   - behavioral score
3. Compute Bayesian-style default probability proxy.
4. Blend all outputs into final confidence score.
5. Generate `APPROVE / HOLD / REJECT` + textual explanations.

## Demo Pointers for Judges

- Use scenarios from `utils/sample_payloads.json`.
- Show:
  - raw inputs
  - decision and confidence
  - risk score
  - explanation bullets
- Emphasize:
  - **Explainability**
  - **Hybrid AI**
  - **Multi-domain adaptability** (finance, healthcare triage, admissions)


## Streamlit Cloud Deployment (No Backend Needed)

This repo now includes a root-level `app.py` so you can deploy directly on Streamlit Cloud without running FastAPI separately.

1. Push your repository to GitHub.
2. Open <https://streamlit.io/cloud> and sign in.
3. Create a new app and select this repo.
4. Set the entry point to `app.py`.
5. Deploy.

The deployed app will show:
- Input form
- Decision output (`APPROVE` / `HOLD` / `REJECT`)
- Risk score
- Explanation bullets

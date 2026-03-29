from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.decision_engine import SmartDecisionEngine


app = FastAPI(title="Smart Decision Engine", version="1.0.0")
engine = SmartDecisionEngine()


class DecisionRequest(BaseModel):
    income: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    liabilities: float = Field(..., ge=0)
    expense_ratio: float = Field(..., ge=0, le=1)
    behavior_score: float = Field(..., ge=0, le=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/decision")
def decision(request: DecisionRequest) -> dict:
    result = engine.evaluate(request.model_dump())
    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "risk_score": result.risk_score,
        "explanation": result.explanation,
        "model_outputs": result.model_outputs,
    }

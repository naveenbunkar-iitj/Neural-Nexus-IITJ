from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DecisionResult:
    decision: str
    confidence: float
    risk_score: float
    explanation: List[str]
    model_outputs: Dict[str, float]


class SmartDecisionEngine:
    """Hybrid AI decision engine for loan/investment risk decisions."""

    def __init__(self) -> None:
        self.tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self._train_tree_model()

    def _train_tree_model(self) -> None:
        """Train on synthetic data suitable for hackathon demos."""
        rng = np.random.default_rng(42)
        rows = 800
        income = rng.normal(70000, 25000, rows).clip(15000, 250000)
        credit_score = rng.normal(670, 80, rows).clip(300, 850)
        liabilities = rng.normal(28000, 18000, rows).clip(0, 180000)
        expense_ratio = rng.uniform(0.2, 0.95, rows)
        behavior = rng.uniform(0.0, 1.0, rows)

        debt_to_income = liabilities / np.maximum(income, 1)

        risk = (
            0.45 * (1 - (credit_score - 300) / 550)
            + 0.25 * debt_to_income.clip(0, 2)
            + 0.20 * expense_ratio
            + 0.10 * (1 - behavior)
        )
        y = (risk < 0.45).astype(int)

        x = np.column_stack([income, credit_score, liabilities, expense_ratio, behavior])
        self.tree_model.fit(x, y)

    @staticmethod
    def _normalized_risk(payload: Dict[str, float]) -> float:
        income = max(payload["income"], 1)
        credit_score = min(max(payload["credit_score"], 300), 850)
        liabilities = max(payload["liabilities"], 0)
        expense_ratio = min(max(payload["expense_ratio"], 0), 1)
        behavior_score = min(max(payload["behavior_score"], 0), 1)

        debt_to_income = liabilities / income
        credit_risk = 1 - ((credit_score - 300) / 550)

        risk = (
            0.4 * credit_risk
            + 0.3 * min(debt_to_income, 2)
            + 0.2 * expense_ratio
            + 0.1 * (1 - behavior_score)
        )
        return float(np.clip(risk, 0, 1))

    @staticmethod
    def _bayesian_probability(payload: Dict[str, float], risk: float) -> float:
        """Simple Bayesian-style likelihood fusion for explainability."""
        credit = payload["credit_score"]
        dti = payload["liabilities"] / max(payload["income"], 1)

        p_default_given_credit = 0.75 if credit < 580 else 0.45 if credit < 680 else 0.15
        p_default_given_dti = 0.75 if dti > 0.65 else 0.45 if dti > 0.4 else 0.2
        p_default_given_behavior = 0.7 if payload["behavior_score"] < 0.35 else 0.25

        combined = (0.45 * p_default_given_credit) + (0.35 * p_default_given_dti) + (0.20 * p_default_given_behavior)
        posterior = 0.6 * combined + 0.4 * risk
        return float(np.clip(1 - posterior, 0, 1))

    def evaluate(self, payload: Dict[str, float]) -> DecisionResult:
        features = np.array(
            [[
                payload["income"],
                payload["credit_score"],
                payload["liabilities"],
                payload["expense_ratio"],
                payload["behavior_score"],
            ]]
        )

        tree_prob = float(self.tree_model.predict_proba(features)[0][1])
        risk_score = self._normalized_risk(payload)
        bayesian_conf = self._bayesian_probability(payload, risk_score)

        final_score = float(np.clip(0.45 * tree_prob + 0.35 * bayesian_conf + 0.20 * (1 - risk_score), 0, 1))

        if final_score >= 0.7:
            decision = "APPROVE"
        elif final_score >= 0.45:
            decision = "HOLD"
        else:
            decision = "REJECT"

        explanations = self._explain(payload, risk_score, decision)

        return DecisionResult(
            decision=decision,
            confidence=round(final_score * 100, 2),
            risk_score=round(risk_score * 100, 2),
            explanation=explanations,
            model_outputs={
                "decision_tree_probability": round(tree_prob, 4),
                "bayesian_probability": round(bayesian_conf, 4),
                "final_score": round(final_score, 4),
            },
        )

    @staticmethod
    def _explain(payload: Dict[str, float], risk_score: float, decision: str) -> List[str]:
        msgs: List[str] = []
        dti = payload["liabilities"] / max(payload["income"], 1)

        if payload["credit_score"] < 620:
            msgs.append("Low credit score increases risk.")
        elif payload["credit_score"] > 740:
            msgs.append("Strong credit profile supports approval.")

        if dti > 0.55:
            msgs.append("High debt-to-income ratio is a major rejection signal.")
        elif dti < 0.3:
            msgs.append("Low debt burden improves repayment confidence.")

        if payload["behavior_score"] < 0.4:
            msgs.append("Behavioral stability is weak based on transaction patterns.")
        else:
            msgs.append("Behavioral pattern suggests stable financial habits.")

        if risk_score > 0.6:
            msgs.append("Overall calculated risk is high.")
        else:
            msgs.append("Overall risk is within acceptable threshold.")

        msgs.append(f"Final recommendation: {decision}.")
        return msgs

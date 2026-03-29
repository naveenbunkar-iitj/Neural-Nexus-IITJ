from fastapi import FastAPI
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# Training dummy model
X = [
    [50000, 700, 10000],
    [30000, 500, 40000],
    [80000, 750, 5000],
    [20000, 400, 30000],
]

y = ["APPROVED", "REJECTED", "APPROVED", "REJECTED"]

model = DecisionTreeClassifier()
model.fit(X, y)


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
def predict(income: int, credit: int, debt: int):
    result = model.predict([[income, credit, debt]])

    score = 0
    if credit > 700:
        score += 2
    if credit < 600:
        score -= 2
    if income > 60000:
        score += 2
    if debt > 30000:
        score -= 2

    return {"decision": result[0], "risk_score": score}

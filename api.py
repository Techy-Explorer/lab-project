from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models safely
try:
    basic_model = joblib.load("expense_model.pkl")
    print("✅ Loaded: expense_model.pkl")
except:
    basic_model = None
    print("❌ ERROR: expense_model.pkl not found")

try:
    advanced_model = joblib.load("expense_predictor_model.pkl")
    print("✅ Loaded: expense_predictor_model.pkl")
except:
    advanced_model = None
    print("❌ ERROR: expense_predictor_model.pkl not found")


# --------------------------
# HEALTH CHECK
# --------------------------
@app.get("/")
def home():
    return {"message": "Full Expense Prediction API running successfully!"}


# --------------------------
# BASIC MODEL ROUTE
# --------------------------
@app.post("/predict-expense-basic")
def predict_basic(amount: float):
    if basic_model is None:
        return {"error": "Basic model not loaded"}

    result = basic_model.predict([[amount]])[0]
    return {"predicted_expense": float(result)}


# --------------------------
# ADVANCED MODEL ROUTE
# --------------------------
@app.post("/predict-expense-advanced")
def predict_advanced(x: float, y: float):
    if advanced_model is None:
        return {"error": "Advanced model not loaded"}

    result = advanced_model.predict([[x, y]])[0]
    return {"predicted_expense": float(result)}


# --------------------------
# FULL 12-FEATURE MODEL ROUTE
# --------------------------
class FullExpenseInput(BaseModel):
    date: str
    mode: str
    category: str
    subcategory: str
    note: str
    amount: float
    type: str
    currency: str
    anomaly: int
    day: int
    month: int
    year: int


@app.post("/predict-expense-full")
def predict_full(data: FullExpenseInput):

    if advanced_model is None:
        return {"error": "Advanced model not loaded"}

    # Currently using the meaningful numerical features
    X = np.array([[data.amount, data.day, data.month, data.year]])

    prediction = advanced_model.predict(X)[0]

    return {
        "predicted_expense": float(prediction),
        "used_features": {
            "amount": data.amount,
            "day": data.day,
            "month": data.month,
            "year": data.year
        }
    }

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# -----------------------------
# 1. FIX PATH (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend folder
CSV_PATH = os.path.join(BASE_DIR, "data", "expenses_pkr.csv")

print("Looking for CSV at:", CSV_PATH)

# -----------------------------
# 2. LOAD CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
print("Loaded Columns:", df.columns.tolist())

# -----------------------------
# 3. CLEAN DATA
# -----------------------------
# Convert Amount to numeric
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

# Only keep rows that are expenses, not income
df = df[df["Income/Expense"].str.lower() == "expense"]

df = df.dropna(subset=["Amount"])

# If no valid data available
if df.empty:
    print("❌ ERROR: No valid expense rows found.")
    exit()

print(f"Total usable expense rows: {len(df)}")

# -----------------------------
# 4. TRAIN BASIC MODEL
# Predict expense using only the amount itself (identity model)
# -----------------------------
try:
    X_basic = df[["Amount"]]
    y_basic = df["Amount"]

    basic_model = LinearRegression()
    basic_model.fit(X_basic, y_basic)

    # Save
    basic_model_path = os.path.join(BASE_DIR, "expense_model.pkl")
    pickle.dump(basic_model, open(basic_model_path, "wb"))

    print("✅ BASIC MODEL TRAINED & SAVED:", basic_model_path)

except Exception as e:
    print("❌ BASIC MODEL TRAINING FAILED:", e)

# -----------------------------
# 5. TRAIN ADVANCED MODEL
# Uses Mode + Category + Subcategory + Amount to predict spending
# -----------------------------
try:
    df_adv = df.copy()

    # Encode strings → numeric
    for col in ["Mode", "Category", "Subcategory"]:
        df_adv[col] = df_adv[col].astype("category").cat.codes

    X_adv = df_adv[["Mode", "Category", "Subcategory", "Amount"]]
    y_adv = df_adv["Amount"]

    adv_model = LinearRegression()
    adv_model.fit(X_adv, y_adv)

    # Save
    adv_model_path = os.path.join(BASE_DIR, "expense_predictor_model.pkl")
    pickle.dump(adv_model, open(adv_model_path, "wb"))

    print("✅ ADVANCED MODEL TRAINED & SAVED:", adv_model_path)

except Exception as e:
    print("❌ ADVANCED MODEL TRAINING FAILED:", e)

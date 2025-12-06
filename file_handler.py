import pandas as pd
from io import StringIO

def preprocess_uploaded_file(file):
    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))

    # Convert "Date" to datetime and extract features
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Day"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
    else:
        df["Day"] = df["Month"] = df["Year"] = None

    return df

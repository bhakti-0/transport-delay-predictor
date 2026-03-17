import joblib
import pandas as pd
import os
from src.preprocess import preprocess_data

def predict_with_data(input_data):
    if not os.path.exists("model.pkl"):
        return None, None

    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")

    df = pd.DataFrame([input_data])
    df = preprocess_data(df)

    df = pd.get_dummies(df)

    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_df], axis=1)
        
    df = df.copy()

    df = df[columns]

    prediction = model.predict(df)[0]

    return prediction, df
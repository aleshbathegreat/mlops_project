import os

import pandas as pd

RAW = "data/Admission_Predict.csv"
OUT = "data/processed.csv"


def prepare_data(raw_path=RAW, out_path=OUT, threshold=0.75):
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Serial No."])

    assert len(df) > 0, "Empty dataset"
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert df["GRE Score"].between(260, 340).all(), "GRE out of range"
    assert df["TOEFL Score"].between(0, 120).all(), "TOEFL out of range"
    assert df["CGPA"].between(0, 10).all(), "CGPA out of range"

    df["target"] = (df["Chance of Admit"] >= threshold).astype(int)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Data ready:", len(df), "rows")
    return df


if __name__ == "__main__":
    prepare_data()

import argparse
import os

import pandas as pd

RAW = "data/Large_Dataset.csv"
OUT = "data/processed.csv"
FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]


def prepare_data(raw_path=RAW, out_path=OUT, threshold=0.75):
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Serial No.", "Motivation Letter"], errors="ignore")
    df = df.drop_duplicates(subset=FEATURES).reset_index(drop=True)

    assert len(df) > 0, "Empty dataset"
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert df["GRE Score"].between(260, 340).all(), "GRE out of range"
    assert df["TOEFL Score"].between(0, 120).all(), "TOEFL out of range"
    assert df["CGPA"].between(0, 10).all(), "CGPA out of range"

    df["target"] = (df["Chance of Admit"] >= threshold).astype(int)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Data ready:", len(df), "rows")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and validate dataset.")
    parser.add_argument("--raw-path", default=RAW, help="Path to raw CSV dataset.")
    parser.add_argument("--out-path", default=OUT, help="Path to processed CSV output.")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="Admit threshold."
    )
    args = parser.parse_args()
    prepare_data(
        raw_path=args.raw_path, out_path=args.out_path, threshold=args.threshold
    )

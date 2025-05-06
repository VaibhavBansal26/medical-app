# eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

def main():
    df = pd.read_csv("data/cleaned_real.csv")
    print("► Shape:", df.shape)
    print("► Columns:", df.columns.tolist())
    print("\n► Summary statistics:\n", df.describe())

    for col in df.select_dtypes(include=["float64","int64"]).columns:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"reports/{col}_hist.png")
        plt.close()
        print(f"Saved histogram for {col} → reports/{col}_hist.png")

if __name__ == "__main__":
    main()

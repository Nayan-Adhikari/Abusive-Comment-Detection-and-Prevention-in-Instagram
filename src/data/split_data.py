"""
Dataset splitting script
Project: Abusive Comment Detection and Prevention

This script:
- Loads cleaned Hinglish dataset
- Performs stratified train/val/test split
- Saves split CSV files
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
INPUT_PATH = "data/processed/hinglish_clean.csv"
OUTPUT_DIR = "data/splits"
RANDOM_STATE = 42
# ---------------------------------------

def create_splits():
    print("[INFO] Loading cleaned dataset...")
    df = pd.read_csv(INPUT_PATH)

    if "label" not in df.columns or "comment" not in df.columns:
        raise ValueError("Dataset must contain 'comment' and 'label' columns")

    print("[INFO] Total samples:", len(df))
    print("[INFO] Class distribution:\n", df["label"].value_counts())

    # 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    # Split temp into 10% validation, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save files
    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print("\n[INFO] Dataset split completed successfully!")
    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    print("\n[INFO] Train class distribution:\n", train_df["label"].value_counts())
    print("\n[INFO] Validation class distribution:\n", val_df["label"].value_counts())
    print("\n[INFO] Test class distribution:\n", test_df["label"].value_counts())


if __name__ == "__main__":
    create_splits()

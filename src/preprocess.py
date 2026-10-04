import os
import glob
import pandas as pd
import numpy as np


# Dataset location
DATA_FOLDER = r"data\MachineLearningCSV\MachineLearningCVE"

# Output location
OUTPUT_FILE = r"data\processed_cicids2017.csv"


def map_attack_label(label):
    """Convert CICIDS2017 raw labels into broader attack families."""

    label = str(label).strip()

    if label == "BENIGN":
        return "BENIGN"

    if label in ["FTP-Patator", "SSH-Patator"]:
        return "BRUTE_FORCE"

    if label.startswith("DoS"):
        return "DOS"

    if label == "DDoS":
        return "DDOS"

    if label == "PortScan":
        return "PORTSCAN"

    if label == "Bot":
        return "BOT"

    if label.startswith("Web Attack"):
        return "WEB_ATTACK"

    if label == "Infiltration":
        return "INFILTRATION"

    if label == "Heartbleed":
        return "HEARTBLEED"

    return "OTHER"


def process_file(file_path):
    """Read and clean one CICIDS2017 CSV file."""

    print(f"\nProcessing: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path, low_memory=False)

    print("Original shape:", df.shape)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Convert raw labels to attack families
    df["Attack_Type"] = df["Label"].apply(map_attack_label)

    # Remove the original raw label
    df = df.drop(columns=["Label"])

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows containing missing values
    before = len(df)
    df = df.dropna()
    removed = before - len(df)

    print("Rows removed:", removed)
    print("Processed shape:", df.shape)

    return df


def main():

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))

    print("CICIDS2017 Preprocessing")
    print("========================")
    print("Files found:", len(files))

    processed_parts = []

    for file_path in files:
        df = process_file(file_path)
        processed_parts.append(df)

    print("\nCombining processed files...")

    final_df = pd.concat(processed_parts, ignore_index=True)

    print("Final shape:", final_df.shape)

    print("\nAttack family distribution:")
    print(final_df["Attack_Type"].value_counts())

    print("\nSaving processed dataset...")

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessing completed successfully.")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
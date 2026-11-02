
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Locate project files
PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_DIR / "data" / "processed" / "matchup_features.csv"
)

# Load prepared basketball matchup data
df = pd.read_csv(FEATURE_FILE)

# Split data chronologically by season
# Older seasons are used for development.
# Later seasons are reserved for testing.

TRAIN_END_SEASON = 2022
TEST_START_SEASON = 2023

train = df[df["Season"] <= TRAIN_END_SEASON].copy()

test = df[df["Season"] >= TEST_START_SEASON].copy()

# Baseline prediction:
# Predict Team A if its prior win percentage
# is greater than or equal to Team B's.
# Otherwise, predict Team B.

test["Prediction"] = (
    test["WinPctDiff"] >= 0
).astype(int)

# Calculate accuracy
accuracy = accuracy_score(
    test["TeamAWon"],
    test["Prediction"]
)

# Display results
print("\n===== BASELINE MODEL RESULTS =====")

print(f"Training matchups: {len(train):,}")
print(f"Testing matchups: {len(test):,}")

print(f"\nBaseline Accuracy: {accuracy:.4f}")
print(f"Baseline Accuracy: {accuracy * 100:.2f}%")

print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        test["TeamAWon"],
        test["Prediction"]
    )
)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        test["TeamAWon"],
        test["Prediction"],
        zero_division=0
    )
)

# Count games where both teams had equal win percentages
ties = (test["WinPctDiff"] == 0).sum()

print(f"\nWin percentage ties: {ties:,}")

print("\nBaseline evaluation completed!")

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# This project uses a simple rule as the first model.
# It does not need scikit-learn for training, only for evaluation.
PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_DIR / "data" / "processed" / "matchup_features.csv"
)

# Load prepared basketball matchup data
df = pd.read_csv(FEATURE_FILE)

# Use older seasons for training and later seasons for testing.
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

# Compare the rule's predictions with the actual outcomes.
accuracy = accuracy_score(
    test["TeamAWon"],
    test["Prediction"]
)

# A tie means Team A and Team B had the same win percentage.
# The rule predicts Team A for every tie.
ties = int((test["WinPctDiff"] == 0).sum())

# Save baseline evaluation results
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "model_results.json"

results = {
    "evaluation": {
        "train_end_season": TRAIN_END_SEASON,
        "test_start_season": TEST_START_SEASON,
        "training_matchups": len(train),
        "test_matchups": len(test)
    },
    "baseline": {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(
            test["TeamAWon"], test["Prediction"]
        ).tolist(),
        "ties": int(ties)
    }
}

with open(RESULTS_FILE, "w") as file:
    json.dump(results, file, indent=4)

print("\nBaseline results saved!")

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

print(f"\nWin percentage ties: {ties:,}")

print("\nBaseline evaluation completed!")

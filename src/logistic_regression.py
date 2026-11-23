
import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
    roc_auc_score
)

# Logistic regression is a first machine-learning model.
# The scaler is included in a pipeline so training and testing
# use the same feature scaling step.
PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_DIR / "data" / "processed" / "matchup_features.csv"
)

# Load prepared matchup dataset
df = pd.read_csv(FEATURE_FILE)

# Choose the three features and the outcome column.
feature_columns = [
    "WinPctDiff",
    "PointsScoredDiff",
    "PointsAllowedDiff"
]

target_column = "TeamAWon"

# Split the data by season instead of randomly mixing games.
TRAIN_END_SEASON = 2022
TEST_START_SEASON = 2023

train = df[df["Season"] <= TRAIN_END_SEASON].copy()
test = df[df["Season"] >= TEST_START_SEASON].copy()

X_train = train[feature_columns]
y_train = train[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

# Standardize the inputs and fit logistic regression.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Predict the outcome and the probability for each test matchup.
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Compare predictions with the actual outcomes.
accuracy = accuracy_score(y_test, predictions)
brier = brier_score_loss(y_test, probabilities)
log_loss_value = log_loss(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)

# Save the same results as JSON so they can be reused by the plots.
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "model_results.json"

existing_results = {}
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, "r") as file:
        existing_results = json.load(file)

existing_results["logistic_regression"] = {
    "accuracy": accuracy,
    "brier_score": brier,
    "log_loss": log_loss_value,
    "roc_auc": roc_auc,
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist()
}

with open(RESULTS_FILE, "w") as file:
    json.dump(existing_results, file, indent=4)

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print(f"Training matchups: {len(train):,}")
print(f"Testing matchups: {len(test):,}")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Brier Score: {brier:.4f}")
print(f"Log Loss: {log_loss_value:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n===== CONFUSION MATRIX =====")

print(confusion_matrix(y_test, predictions))

print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\n===== SAMPLE PREDICTIONS =====")

sample = test[feature_columns].head(5).copy()

sample["Actual"] = y_test.head(5).values
sample["Predicted"] = predictions[:5]
sample["Team A Win Probability"] = probabilities[:5]

print(sample.to_string(index=False))

print("\nLogistic regression evaluation completed!")

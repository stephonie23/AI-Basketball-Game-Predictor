
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

# Locate project files
PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_DIR / "data" / "processed" / "matchup_features.csv"
)

# Load prepared matchup dataset
df = pd.read_csv(FEATURE_FILE)

# Define input features and prediction target
feature_columns = [
    "WinPctDiff",
    "PointsScoredDiff",
    "PointsAllowedDiff"
]

target_column = "TeamAWon"

# Chronological split
TRAIN_END_SEASON = 2022
TEST_START_SEASON = 2023

train = df[df["Season"] <= TRAIN_END_SEASON].copy()
test = df[df["Season"] >= TEST_START_SEASON].copy()

X_train = train[feature_columns]
y_train = train[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

# Build logistic regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Generate predicted probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
brier = brier_score_loss(y_test, probabilities)

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print(f"Training matchups: {len(train):,}")
print(f"Testing matchups: {len(test):,}")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Brier Score: {brier:.4f}")

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

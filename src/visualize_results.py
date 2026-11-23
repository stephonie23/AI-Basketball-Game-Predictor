
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURE_FILE = PROJECT_DIR / "data" / "processed" / "matchup_features.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load matchup data
df = pd.read_csv(FEATURE_FILE)

# Graph 1: compare the baseline with logistic regression.
models = ["Baseline", "Logistic Regression"]
with open(RESULTS_DIR / "model_results.json", "r") as file:
    model_results = json.load(file)

accuracies = [
    model_results["baseline"]["accuracy"] * 100,
    model_results["logistic_regression"]["accuracy"] * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

plt.title("Basketball Prediction Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_accuracy.png",
    dpi=300
)

plt.close()

# Graph 2: show how many prepared matchups Team A and Team B won.
win_counts = df["TeamAWon"].value_counts().sort_index()

labels = ["Team B Wins", "Team A Wins"]
values = [
    win_counts.get(0, 0),
    win_counts.get(1, 0)
]

plt.figure(figsize=(8, 5))

plt.bar(labels, values)

plt.title("Distribution of Women's Basketball Game Outcomes")
plt.ylabel("Number of Games")

for i, value in enumerate(values):
    plt.text(i, value + 500, f"{value:,}", ha="center")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "win_distribution.png",
    dpi=300
)

plt.close()

print("\n===== VISUALIZATIONS CREATED =====")
print("1. results/model_accuracy.png")
print("2. results/win_distribution.png")
print("\nVisualization process completed!")

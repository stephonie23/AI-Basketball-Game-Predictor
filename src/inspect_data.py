
import pandas as pd
from pathlib import Path

# Locate the project data folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Load women's basketball datasets
compact = pd.read_csv(DATA_DIR / "WRegularSeasonCompactResults.csv")
detailed = pd.read_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv")
teams = pd.read_csv(DATA_DIR / "WTeams.csv")
seasons = pd.read_csv(DATA_DIR / "WSeasons.csv")

# Display dataset information
print("\n===== DATASET OVERVIEW =====")

print(f"Compact game records: {len(compact):,}")
print(f"Detailed game records: {len(detailed):,}")
print(f"Number of teams: {len(teams):,}")
print(f"Number of seasons: {len(seasons):,}")

print("\n===== SEASONS =====")
print(seasons.head())

print("\n===== TEAM INFORMATION =====")
print(teams.head())

print("\n===== COMPACT GAME DATA =====")
print(compact.head())

print("\n===== DETAILED GAME COLUMNS =====")
print(detailed.columns.tolist())

print("\n===== MISSING VALUES =====")
print(compact.isnull().sum())
print(detailed.isnull().sum())

print("\nDataset inspection completed!")

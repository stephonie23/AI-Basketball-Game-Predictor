# AI Basketball Game Predictor — NCAA Women's Basketball

## Project Overview

This CSC 425 Artificial Intelligence project explores whether historical NCAA Division I women's basketball results and team statistics can predict the winner of a future matchup.

The project uses historical game data to develop and evaluate machine-learning models. The models use information available before each game to predict the winner of a matchup.

## Problem Definition

- **Input:** Two women's college basketball teams and their historical performance statistics.
- **AI Task:** Supervised binary classification.
- **Output:** Predicted winner and estimated win probability when supported by the model.
- **Goal:** Compare a simple baseline model with machine-learning approaches using historical games held out from training.

## Dataset

**Primary Source:** [Kaggle — March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)

The project uses the following women's basketball datasets:

- `WRegularSeasonCompactResults.csv` — Historical regular-season game results.
- `WRegularSeasonDetailedResults.csv` — Detailed team box-score statistics.
- `WTeams.csv` — Women's team IDs and names.
- `WSeasons.csv` — Season information.

The dataset contains women's basketball seasons from 1998 through 2026.

Downloaded CSV files are stored in `data/raw/` and are not committed to this repository.

## Data Preparation and Feature Engineering

Historical game results are converted into matchup records containing pregame team statistics.

The following features are calculated:

- Win percentage difference.
- Average points scored difference.
- Average points allowed difference.

Team A is consistently assigned as the team with the lower team ID, while Team B has the higher team ID. The target variable identifies whether Team A won.

To prevent data leakage:
- Features only use information available before the game being predicted.
- Games played on the same day are processed together before updating team histories.
- Game results are not used as input features for predicting that same game.
- Teams must have at least one previous game in the season to be included.

The feature verification script independently checks the calculations.

**Data preparation results:**
- Original games: 142,507
- Skipped games: 5,994
- Usable matchups: 136,513
- Feature verification errors: 0

## Models and Evaluation

The project currently includes two models:

### 1. Baseline Model

The baseline predicts Team A when the win percentage difference is greater than or equal to zero. Otherwise, it predicts Team B.

**Accuracy: 70.90%**

### 2. Logistic Regression

Logistic regression uses the three historical performance differences:

- Win percentage difference.
- Average points scored difference.
- Average points allowed difference.

The model uses StandardScaler for feature scaling and LogisticRegression for classification.

**Results:**
- Accuracy: 73.08%
- Brier score: 0.1794

### Training and Testing Strategy

The models use a chronological split by season:

- Training: Seasons through 2022.
- Testing: Seasons 2023 through 2026.

Both models were evaluated on the same 20,824 test matchups.

The 2026 season data may be incomplete depending on the dataset snapshot.

## Project Structure

```text
AI-Basketball-Game-Predictor/
├── data/
│   ├── raw/                 # Downloaded Kaggle datasets
│   └── processed/
│       └── matchup_features.csv
├── notebooks/
├── results/
│   ├── model_accuracy.png
│   └── win_distribution.png
├── src/
│   ├── inspect_data.py
│   ├── build_features.py
│   ├── verify_features.py
│   ├── baseline_model.py
│   ├── logistic_regression.py
│   └── visualize_results.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### Requirements

- Python 3.12
- pip
- Kaggle women's basketball dataset files

### 1. Clone the Repository

```bash
git clone https://github.com/stephonie23/AI-Basketball-Game-Predictor.git
cd AI-Basketball-Game-Predictor
```

### 2. Create and Activate a Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Dataset

Download the required women's basketball CSV files from Kaggle and place them inside `data/raw/`.

### 5. Run the Project

Run the scripts from the main project directory:

```bash
python src/inspect_data.py
python src/build_features.py
python src/verify_features.py
python src/baseline_model.py
python src/logistic_regression.py
python src/visualize_results.py
```

The feature-generation script creates the processed matchup dataset. The model scripts evaluate predictions, and the visualization script generates the result graphs.

## Results and Visualizations

The project generates two visualizations:

- `results/model_accuracy.png` — Compares baseline and logistic regression accuracy.
- `results/win_distribution.png` — Shows the distribution of Team A and Team B wins in the prepared dataset.

## Current Project Status

- **Milestone 1:** Project proposal, dataset selection, and repository setup — Completed.
- **Milestone 2:** Data preparation, feature engineering, verification, baseline model, and logistic regression — Completed.

## Future Work

Future development may include:

- Comparing additional machine-learning models, such as Random Forest.
- Analyzing incorrect predictions.
- Exploring additional team performance statistics.
- Improving model performance and probability estimates.
- Developing a demonstration that predicts the winner of a selected matchup.

## Team

**Team Member:** Stephonie Cole

**Course:** CSC 425 — Artificial Intelligence

**Project Type:** Individual Project

## References

1. Kaggle. (2026). [March Machine Learning Mania 2026 — Data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data).
2. NCAA. [Women's Basketball Statistics and Records](https://www.ncaa.org/championships/statistics-and-records/womens-basketball/).
3. scikit-learn developers. [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
4. scikit-learn developers. [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

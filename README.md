# AI Basketball Game Predictor --- NCAA Women's Basketball

## Project Overview

This CSC 425 Artificial Intelligence project explores whether historical
NCAA Division I women's basketball results and team statistics can
predict the winner of a future matchup. The model will output a
predicted winner and, if supported by the selected model, estimated win
probabilities.

## Problem Definition

-   **Input:** Two women's college basketball teams and information
    available before tipoff, including prior win percentage, points
    scored/allowed, selected box-score averages, season/date, and venue
    when available.
-   **AI task:** Supervised binary classification.
-   **Output:** Predicted winner and estimated win probability.
-   **Goal:** Compare a simple baseline with one or more
    machine-learning models using historical games held out from
    training.

## Dataset

Primary source: [Kaggle --- March Machine Learning Mania
2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)

Use the women's data files only: - `WRegularSeasonCompactResults.csv`
--- historical regular-season results; women's data begins in the 1998
season. - `WRegularSeasonDetailedResults.csv` --- team box-score
statistics; women's data begins in the 2010 season. - `WTeams.csv` ---
women's team IDs and names. - `WSeasons.csv` --- season information.

The detailed-results files contain statistics such as shooting,
rebounds, assists, steals, blocks, and turnovers. Some detailed women's
games are missing in the 2010--2012 seasons; detailed results are
available for all games from 2013 onward according to the dataset
documentation.

**Data access:** Download the files from Kaggle and place them in
`data/raw/`. Do not commit downloaded data to this repository. Follow
Kaggle's current competition rules and terms.

## Proposed Method

1.  Inspect the data and check types, duplicates, and missing values.
2.  Convert each completed game into a consistent two-team matchup row.
3.  Build pregame features using only information available before each
    game (for example, rolling or season-to-date averages).
4.  Set the target label to indicate which designated team won.
5.  Train a simple baseline and logistic regression; compare a random
    forest if time permits.
6.  Use a chronological split by season so later seasons test the model
    on games it has not seen.
7.  Evaluate with accuracy and, when probability estimates are
    available, Brier score. Analyze incorrect predictions and
    limitations.

**Leakage prevention:** Do not use a game's final score or box-score
statistics as input features for predicting that same game. Those values
can only update a team's historical features after the game has
occurred.

## Initial Project Structure

``` text
womens-basketball-game-predictor/
├── README.md
├── data/
│   └── raw/              # Downloaded data (not committed)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── reports/
│   └── figures/
├── requirements.txt
└── .gitignore
```

## Team and Communication

-   **Team members:** Stephonie Cole
-   **Initial responsibilities:** Individual project
## Project Timeline

  -----------------------------------------------------------------------
  Period                              Target
  ----------------------------------- -----------------------------------
  Oct. 11--18, 2026                   Finalize scope, team
                                      responsibilities, repository, and
                                      inspect data

  Oct. 19--25                         Preprocess data, create pregame
                                      features, build baseline

  Oct. 26--Nov. 15                    Train and compare models

  Nov. 16--29                         Evaluate on later seasons, analyze
                                      errors, refine

  Nov. 30--Dec. 11                    Final report, reproducible code,
                                      demo, presentation
  -----------------------------------------------------------------------

## Expected Final Deliverable

A reproducible Python project with preprocessing and feature-generation
code, a trained classifier, evaluation results, and a notebook or small
demonstration that predicts a winner/probability for a selected NCAA
women's basketball matchup.

## Initial References

1.  Kaggle. (2026). [March Machine Learning Mania 2026 ---
    Data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data).
2.  Kaggle. (2026). [March Machine Learning Mania
    2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/).
3.  NCAA. [Women's Basketball Statistics &
    Records](https://www.ncaa.org/championships/statistics-and-records/womens-basketball/).
4.  scikit-learn developers. [Model evaluation: quantifying the quality
    of
    predictions](https://scikit-learn.org/stable/modules/model_evaluation.html).
5.  scikit-learn developers. [Logistic
    regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

## Status

Milestone 1: proposal, data source selection, and repository setup. A
trained model is not required for this milestone.

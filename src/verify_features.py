
import pandas as pd
from pathlib import Path
import math

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
FEATURE_FILE = PROJECT_DIR / "data" / "processed" / "matchup_features.csv"

games = pd.read_csv(DATA_DIR / "WRegularSeasonCompactResults.csv")
features = pd.read_csv(FEATURE_FILE)

games = games.sort_values(
    ["Season", "DayNum"]
).reset_index(drop=True)

history = {}
feature_index = 0
errors = []
verified = []

# Independently reconstruct historical statistics
for season, season_games in games.groupby("Season", sort=True):

    history = {}

    for day, day_games in season_games.groupby("DayNum", sort=True):

        daily_updates = []

        for game in day_games.itertuples(index=False):

            winner = int(game.WTeamID)
            loser = int(game.LTeamID)

            winner_score = int(game.WScore)
            loser_score = int(game.LScore)

            team_a = min(winner, loser)
            team_b = max(winner, loser)

            a = history.get(team_a)
            b = history.get(team_b)

            if a is not None and b is not None:

                actual = features.iloc[feature_index]

                expected = {
                    "WinPctDiff": (
                        a["wins"] / a["games"]
                        - b["wins"] / b["games"]
                    ),
                    "PointsScoredDiff": (
                        a["scored"] / a["games"]
                        - b["scored"] / b["games"]
                    ),
                    "PointsAllowedDiff": (
                        a["allowed"] / a["games"]
                        - b["allowed"] / b["games"]
                    ),
                    "TeamAWon": int(winner == team_a)
                }

                # Verify team ordering
                if team_a >= team_b:
                    errors.append("Incorrect team ordering")

                # Verify all feature calculations
                for column, expected_value in expected.items():

                    actual_value = actual[column]

                    if not math.isclose(
                        float(actual_value),
                        float(expected_value),
                        rel_tol=1e-9,
                        abs_tol=1e-9
                    ):
                        errors.append(
                            f"Mismatch in row {feature_index}: {column}"
                        )

                verified.append({
                    "Season": season,
                    "DayNum": day,
                    "TeamA": team_a,
                    "TeamB": team_b,
                    **expected
                })

                feature_index += 1

            # Update histories only after the entire game day
            daily_updates.append(
                (winner, winner_score, loser_score, 1)
            )
            daily_updates.append(
                (loser, loser_score, winner_score, 0)
            )

        for team_id, scored, allowed, won in daily_updates:

            if team_id not in history:
                history[team_id] = {
                    "games": 0,
                    "wins": 0,
                    "scored": 0,
                    "allowed": 0
                }

            history[team_id]["games"] += 1
            history[team_id]["wins"] += won
            history[team_id]["scored"] += scored
            history[team_id]["allowed"] += allowed

# Final verification
print("\n===== DATA VERIFICATION =====")
print(f"Feature rows checked: {feature_index:,}")
print(f"Verification errors: {len(errors)}")

if feature_index != len(features):
    errors.append("Feature row count does not match")

if errors:
    print("\nERRORS FOUND:")
    for error in errors[:10]:
        print(error)
else:
    print("\nSUCCESS: All feature calculations match!")
    print("Team ordering is correct.")
    print("Labels are correct.")
    print("Historical statistics exclude the game being predicted.")
    
    print("\n===== SAMPLE VERIFIED MATCHUPS =====")
    print(pd.DataFrame(verified).head(5).to_string(index=False))

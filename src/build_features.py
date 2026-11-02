
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load historical women's basketball results
games = pd.read_csv(
    DATA_DIR / "WRegularSeasonCompactResults.csv"
)

games = games.sort_values(
    ["Season", "DayNum"]
).reset_index(drop=True)

matchups = []
skipped_games = 0

# Process one season at a time
for season, season_games in games.groupby("Season", sort=True):

    # Store statistics from previously completed games
    team_history = {}

    # Process one game day at a time
    for day, day_games in season_games.groupby("DayNum", sort=True):

        daily_updates = []

        for game in day_games.itertuples(index=False):

            winner_id = int(game.WTeamID)
            loser_id = int(game.LTeamID)

            winner_score = int(game.WScore)
            loser_score = int(game.LScore)

            # Team A is always the lower team ID
            team_a_id = min(winner_id, loser_id)
            team_b_id = max(winner_id, loser_id)

            # Retrieve historical statistics
            team_a = team_history.get(team_a_id)
            team_b = team_history.get(team_b_id)

            # Skip games without prior history for both teams
            if team_a is None or team_b is None:
                skipped_games += 1
            else:

                a_win_pct = team_a["wins"] / team_a["games"]
                b_win_pct = team_b["wins"] / team_b["games"]

                a_points_scored = (
                    team_a["points_scored"] / team_a["games"]
                )
                b_points_scored = (
                    team_b["points_scored"] / team_b["games"]
                )

                a_points_allowed = (
                    team_a["points_allowed"] / team_a["games"]
                )
                b_points_allowed = (
                    team_b["points_allowed"] / team_b["games"]
                )

                matchups.append({
                    "Season": season,
                    "DayNum": day,
                    "TeamA": team_a_id,
                    "TeamB": team_b_id,

                    "WinPctDiff": a_win_pct - b_win_pct,

                    "PointsScoredDiff": (
                        a_points_scored - b_points_scored
                    ),

                    "PointsAllowedDiff": (
                        a_points_allowed - b_points_allowed
                    ),

                    # 1 = Team A won, 0 = Team B won
                    "TeamAWon": int(winner_id == team_a_id)
                })

            # Save results for updating AFTER the entire game day
            daily_updates.append((
                winner_id, winner_score, loser_score, 1
            ))

            daily_updates.append((
                loser_id, loser_score, winner_score, 0
            ))

        # Update team histories after all games that day
        for team_id, scored, allowed, won in daily_updates:

            if team_id not in team_history:
                team_history[team_id] = {
                    "games": 0,
                    "wins": 0,
                    "points_scored": 0,
                    "points_allowed": 0
                }

            team_history[team_id]["games"] += 1
            team_history[team_id]["wins"] += won
            team_history[team_id]["points_scored"] += scored
            team_history[team_id]["points_allowed"] += allowed


# Create the final matchup dataset
features = pd.DataFrame(matchups)

output_file = OUTPUT_DIR / "matchup_features.csv"
features.to_csv(output_file, index=False)

print("\n===== FEATURE BUILDING RESULTS =====")
print(f"Total original games: {len(games):,}")
print(f"Games skipped: {skipped_games:,}")
print(f"Usable matchups: {len(features):,}")

print("\n===== FEATURE DATASET =====")
print(features.head())

print("\n===== LABEL DISTRIBUTION =====")
print(features["TeamAWon"].value_counts())

print("\n===== MISSING VALUES =====")
print(features.isnull().sum())

print(f"\nSaved dataset to: {output_file}")

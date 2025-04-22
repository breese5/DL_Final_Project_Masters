import pandas as pd
import os
from glob import glob
from functools import reduce

def normalize_name(name):
    """
    Normalize a player name from 'Last, First' to 'first last' format in lowercase.

    Args:
        name (str): Name string to normalize.

    Returns:
        str: Normalized name string.
    """
    name = str(name).strip().lower()
    if "," in name:
        parts = [part.strip() for part in name.split(",")]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    return name

def convert_fractional_odds(odds_str):
    """
    Convert fractional odds (e.g., '5/1') into an implied win probability.

    Args:
        odds_str (str): Fractional odds as a string.

    Returns:
        float or None: Probability value or None if conversion fails.
    """
    try:
        numerator, denominator = odds_str.split('/')
        return 1 / (int(numerator) / int(denominator) + 1)
    except:
        return None

def build_features():
    """
    Generate and save a traditional tournament feature dataset for model training.

    - Loads Masters betting odds
    - Collects strokes gained stats from all historical result CSVs
    - Normalizes player names for merging
    - Merges all tournament data into one feature set
    - Combines with odds data and saves to processed folder
    """
    base_path = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_path, ".."))
    raw_path = os.path.join(project_root, "data/raw")
    processed_path = os.path.join(project_root, "data/processed")
    os.makedirs(processed_path, exist_ok=True)

    # Load odds data
    odds_file = os.path.join(raw_path, "MastersOdds2025.csv")
    odds_df = pd.read_csv(odds_file)
    odds_df["Golfer"] = odds_df["Golfer"].apply(normalize_name)
    odds_df["Odds Prob"] = odds_df["Odds"].apply(convert_fractional_odds)

    # Get all CSVs in the raw data folder, excluding the odds file
    all_files = glob(os.path.join(raw_path, "*.csv"))
    tournament_files = [f for f in all_files if "MastersOdds2025.csv" not in f]

    sg_metrics = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"]
    player_metric_data = []

    for file in tournament_files:
        df = pd.read_csv(file)
        filename = os.path.basename(file).replace("_results.csv", "")
        
        # Detect player name column
        name_col = next((col for col in df.columns if "player" in col.lower()), None)
        if not name_col:
            continue

        df["player_name"] = df[name_col].apply(normalize_name)

        # Rename strokes gained metrics with tournament suffix
        for metric in sg_metrics:
            if metric in df.columns:
                df[f"{metric}_{filename}"] = df[metric]

        keep_cols = ["player_name"] + [f"{m}_{filename}" for m in sg_metrics if f"{m}_{filename}" in df.columns]
        player_metric_data.append(df[keep_cols])

    if not player_metric_data:
        raise ValueError("No valid strokes gained data found.")

    # Merge all strokes gained data on player name
    features_df = reduce(lambda left, right: pd.merge(left, right, on="player_name", how="outer"), player_metric_data)

    # Merge strokes gained data with odds
    final_df = pd.merge(odds_df, features_df, left_on="Golfer", right_on="player_name", how="left")

    # Save final dataset
    output_file = os.path.join(processed_path, "traditional_model_10_tournament_data.csv")
    final_df.to_csv(output_file, index=False)
    print(f"✅ Saved combined tournament feature data to {output_file}")

if __name__ == "__main__":
    build_features()

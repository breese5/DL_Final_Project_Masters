import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def normalize_name(name):
    """
    Normalize golfer names to 'first last' format.
    
    Args:
        name (str): Original player name, potentially in 'Last, First' format.
    
    Returns:
        str: Normalized name in lowercase 'first last' format.
    """
    name = str(name).strip().lower()
    if "," in name:
        parts = [part.strip() for part in name.split(",")]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    return name

def convert_fractional_odds(odds_str):
    """
    Convert fractional odds (e.g. '5/1') into implied probability.
    
    Args:
        odds_str (str): Fractional odds as a string.
    
    Returns:
        float or None: Implied probability, or None on failure.
    """
    try:
        numerator, denominator = odds_str.split('/')
        return 1 / (int(numerator) / int(denominator) + 1)
    except:
        return None

def extract_labels_from_results(raw_path):
    """
    Extract target labels (Victory, Top 5, Top 10, Made Cut) from historical tournament result CSVs.
    
    Args:
        raw_path (str): Path to the directory containing raw result files.
    
    Returns:
        pd.DataFrame: Labeled data with players' performance outcomes across tournaments.
    """
    label_data = []

    for fname in os.listdir(raw_path):
        if not fname.endswith(".csv"):
            continue
        if "2025_masters_tournament_results.csv" in fname:
            continue  # skip the future tournament

        df = pd.read_csv(os.path.join(raw_path, fname))
        if "player_name" not in df.columns or "position" not in df.columns:
            continue

        df["player_name"] = df["player_name"].apply(normalize_name)
        df["Made Cut"] = df["position"].apply(lambda x: 0 if str(x).strip().upper() == "CUT" else 1)

        def norm_pos(pos):
            try:
                return int(str(pos).replace("T", ""))
            except:
                return None

        df["position_numeric"] = df["position"].apply(norm_pos)
        df["Victory"] = df["position_numeric"].apply(lambda x: 1 if x == 1 else 0)
        df["Top 5"] = df["position_numeric"].apply(lambda x: 1 if x is not None and x <= 5 else 0)
        df["Top 10"] = df["position_numeric"].apply(lambda x: 1 if x is not None and x <= 10 else 0)

        df["tournament_key"] = fname.replace(".csv", "")
        label_data.append(df[["player_name", "Victory", "Top 5", "Top 10", "Made Cut", "tournament_key"]])

    return pd.concat(label_data, ignore_index=True)

def run_traditional_model():
    """
    Train a traditional Random Forest model using player performance and tournament features.

    - Loads features from processed player tournament data
    - Merges with labels from historical tournaments (excluding 2025)
    - Trains separate Random Forest models for each target category
    - Predicts probabilities and saves results to disk

    Returns:
        pd.DataFrame: Final predictions for each player with target probabilities
    """
    base_path = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_path, ".."))
    data_path = os.path.join(project_root, "data")
    processed_path = os.path.join(data_path, "processed")
    raw_path = os.path.join(data_path, "raw")
    model_path = os.path.join(project_root, "models")
    os.makedirs(model_path, exist_ok=True)

    features_df = pd.read_csv(os.path.join(processed_path, "traditional_model_10_tournament_data.csv"))
    features_df["Golfer"] = features_df["Golfer"].apply(normalize_name)

    if "Odds Prob" not in features_df.columns:
        features_df["Odds Prob"] = features_df["Odds"].apply(convert_fractional_odds)

    label_df = extract_labels_from_results(raw_path)

    merged = pd.merge(features_df, label_df, left_on="Golfer", right_on="player_name", how="inner")
    print(f"✅ Merged training data shape: {merged.shape}")
    if merged.empty:
        print("❌ No matching data after label merge.")
        return

    targets = ["Victory", "Top 5", "Top 10", "Made Cut"]
    non_features = ["Golfer", "Odds", "player_name", "tournament_key"] + targets
    feature_cols = [col for col in merged.columns if col not in non_features and pd.api.types.is_numeric_dtype(merged[col])]
    X = merged[feature_cols].dropna(axis=1, how='all')
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    results = pd.DataFrame()
    results["Golfer"] = merged["Golfer"]
    models = {}

    for target in targets:
        y = merged[target]
        rf = RandomForestClassifier(random_state=42)
        param_grid = {"max_depth": [4, 6, 8], "n_estimators": [100, 200]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_imputed, y)

        best_model = grid.best_estimator_
        prob = best_model.predict_proba(X_imputed)[:, 1]
        results[f"{target} Prob"] = prob
        models[target] = best_model

    # Aggregate repeated entries per player (if any)
    results = results.groupby("Golfer", as_index=False).mean()

    results.to_csv(os.path.join(processed_path, "traditional_predictions.csv"), index=False)
    with open(os.path.join(model_path, "traditional_model.pkl"), "wb") as f:
        pickle.dump(models, f)

    print("✅ Traditional model trained and predictions saved.")
    return results

if __name__ == "__main__":
    run_traditional_model()

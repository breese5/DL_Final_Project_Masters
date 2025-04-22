import pandas as pd
import os
import pickle

def convert_fractional_odds(odds_str):
    """
    Convert fractional betting odds to implied probability.
    
    Args:
        odds_str (str): Fractional odds (e.g., "5/1")
        
    Returns:
        float or None: Implied probability if valid, else None
    """
    try:
        numerator, denominator = odds_str.split('/')
        return 1 / (int(numerator) / int(denominator) + 1)
    except:
        return None

def run_naive_model():
    """
    Execute a naive prediction model using betting odds and OWGR data.

    This model:
    - Converts betting odds to implied probabilities
    - Normalizes victory probabilities
    - Assigns top 5 and top 10 probabilities based on odds rank
    - Assumes all players make the cut
    - Outputs predictions and saves them in both CSV and pickle format

    Returns:
        pd.DataFrame: DataFrame of predictions for each player
    """
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "../data/raw")
    processed_path = os.path.join(base_path, "../data/processed")
    model_path = os.path.join(base_path, "../models")
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Load data
    owgr = pd.read_csv(os.path.join(data_path, "OWGRpreMasters.csv"))
    odds = pd.read_csv(os.path.join(data_path, "MastersOdds2025.csv"))

    # Clean and merge player names
    owgr["Full Name"] = owgr["First Name"].str.strip() + " " + owgr["Last Name"].str.strip()
    odds["Golfer"] = odds["Golfer"].str.strip()

    # Convert and normalize odds
    odds["Victory Prob"] = odds["Odds"].apply(convert_fractional_odds)
    odds.dropna(subset=["Victory Prob"], inplace=True)
    odds["Victory Prob"] = odds["Victory Prob"] / odds["Victory Prob"].sum()

    # Merge datasets
    merged = pd.merge(odds, owgr, how="left", left_on="Golfer", right_on="Full Name")

    # Assign simple top 5 / top 10 / made cut logic
    merged = merged.sort_values("Victory Prob", ascending=False).reset_index(drop=True)
    merged["Top 5 Prob"] = merged.index.map(lambda x: 1.0 if x < 5 else 0.3 if x < 10 else 0.1)
    merged["Top 10 Prob"] = merged.index.map(lambda x: 1.0 if x < 10 else 0.2)
    merged["Made Cut Prob"] = 1.0  # Assumes all make cut

    # Final prediction output
    output = merged[["Golfer", "Victory Prob", "Top 5 Prob", "Top 10 Prob", "Made Cut Prob"]]

    # Save predictions
    output.to_csv(os.path.join(processed_path, "naive_predictions.csv"), index=False)

    # Save dummy model (DataFrame as a pickle)
    with open(os.path.join(model_path, "naive_model.pkl"), "wb") as f:
        pickle.dump(output, f)

    return output

def main():
    """
    CLI entry point to run the naive prediction model.
    """
    print("⚙️  Running naive model...")
    predictions = run_naive_model()
    print("✅ Predictions saved to /data/processed/naive_predictions.csv")
    print(predictions.head())

if __name__ == "__main__":
    main()

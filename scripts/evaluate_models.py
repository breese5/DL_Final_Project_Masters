import pandas as pd
from sklearn.metrics import brier_score_loss, accuracy_score, f1_score, recall_score
import os

def normalize_name(name):
    """
    Normalize a golfer's name to lowercase format "first last".
    Converts "Last, First" to "first last" and strips whitespace.

    Args:
        name (str): The raw name string.

    Returns:
        str: Normalized name in "first last" format.
    """
    name = str(name).lower().strip()
    if "," in name:
        last, first = [x.strip() for x in name.split(",")]
        return f"{first} {last}"
    return name

def preprocess_actual_results(df):
    """
    Convert the position data from the Masters results file into
    binary classification labels for each prediction task.

    Args:
        df (pd.DataFrame): DataFrame with a 'position' column.

    Returns:
        pd.DataFrame: Original DataFrame with new binary label columns:
                      "Victory", "Top 5", "Top 10", "Made Cut"
    """
    def normalize_position(pos):
        try:
            return int(str(pos).replace("T", ""))
        except:
            return None

    df["Made Cut"] = df["position"].apply(lambda x: 0 if str(x).strip().upper() == "CUT" else 1)
    df["position"] = df["position"].apply(normalize_position)
    df["Victory"] = df["position"].apply(lambda x: 1 if x == 1 else 0)
    df["Top 5"] = df["position"].apply(lambda x: 1 if x is not None and x <= 5 else 0)
    df["Top 10"] = df["position"].apply(lambda x: 1 if x is not None and x <= 10 else 0)
    return df

def evaluate_model(pred_file: str, model_name: str):
    """
    Evaluate the predictions of a model using multiple metrics:
    - Brier Score
    - Accuracy @ 0.5 threshold
    - F1 Score @ 0.5 threshold
    - Recall @ 0.5 threshold

    Args:
        pred_file (str): The CSV file containing model predictions.
        model_name (str): Name of the model for display/logging purposes.

    Returns:
        list: A list of dictionaries, each with metric results for one target task.
    """
    base_path = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_path, ".."))
    processed_path = os.path.join(project_root, "data/processed")
    raw_path = os.path.join(project_root, "data/eval")

    preds = pd.read_csv(os.path.join(processed_path, pred_file))
    actual = pd.read_csv(os.path.join(raw_path, "2025_masters_tournament_results.csv"))

    preds["Golfer"] = preds["Golfer"].apply(normalize_name)
    actual["player_name"] = actual["player_name"].apply(normalize_name)
    actual = preprocess_actual_results(actual)

    df = pd.merge(preds, actual, left_on="Golfer", right_on="player_name", how="inner")
    if df.empty:
        print(f"❌ No matching players found for {model_name}")
        return []

    metrics = []
    labels_probs = [
        ("Victory", "Victory Prob"),
        ("Top 5", "Top 5 Prob"),
        ("Top 10", "Top 10 Prob"),
        ("Made Cut", "Made Cut Prob")
    ]

    print(f"🔍 Evaluation Metrics ({model_name}):\n")

    for label, prob in labels_probs:
        if prob not in df.columns or label not in df.columns:
            print(f"{label}: Skipping – column missing.\n")
            continue

        y_true = df[label]
        y_prob = df[prob]

        try:
            brier = brier_score_loss(y_true, y_prob)
            y_pred = (y_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            print(f"{label}:")
            print(f"  Brier Score: {brier:.4f}")
            print(f"  Accuracy @ 0.5 threshold: {acc:.4f}")
            print(f"  F1 Score @ 0.5 threshold:  {f1:.4f}")
            print(f"  Recall @ 0.5 threshold:     {recall:.4f}\n")

            metrics.append({
                "Model": model_name,
                "Target": label,
                "Brier Score": round(brier, 4),
                "Accuracy (0.5)": round(acc, 4),
                "F1 Score (0.5)": round(f1, 4),
                "Recall (0.5)": round(recall, 4)
            })

        except Exception as e:
            print(f"{label}: Error in evaluation → {e}\n")

    return metrics

def evaluate_all_models():
    """
    Evaluate all prediction CSVs (naive, traditional, deep learning)
    and store a combined evaluation summary in data/eval/.
    """
    all_metrics = []
    all_metrics += evaluate_model("naive_predictions.csv", "Naive Model")
    all_metrics += evaluate_model("traditional_predictions.csv", "Traditional Model")
    all_metrics += evaluate_model("deep_learning_predictions.csv", "Deep Learning Model")

    eval_df = pd.DataFrame(all_metrics)

    eval_dir = os.path.join("data", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_df.to_csv(os.path.join(eval_dir, "evaluation_summary.csv"), index=False)

    print("\n✅ Evaluation summary saved to data/eval/evaluation_summary.csv")

if __name__ == "__main__":
    evaluate_all_models()

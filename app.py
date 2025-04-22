import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_dir = os.path.dirname(__file__)
processed_path = os.path.join(base_dir, "data", "processed")
eval_path = os.path.join(base_dir, "data", "eval", "evaluation_summary.csv")

# Load CSV from processed directory
def load_csv(file_name):
    path = os.path.join(processed_path, file_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def show_predictions(title, file_name):
    df = load_csv(file_name)
    if df is not None:
        st.subheader(title)
        st.dataframe(df)
    else:
        st.warning(f"{title} predictions not found.")

def plot_metric_comparison(eval_df):
    st.subheader("📊 Metric Comparison (Bar Chart)")

    # Only numeric metrics
    metric_cols = ["Brier Score", "Accuracy (0.5)", "F1 Score (0.5)", "Recall (0.5)"]

    # Melt into long format for seaborn
    melted = eval_df.melt(
        id_vars=["Model", "Target"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value"
    )

    # Create a combined column to group by Target + Metric
    melted["Metric Group"] = melted["Target"] + " – " + melted["Metric"]

    # Plot with seaborn
    plt.figure(figsize=(14, 6))
    sns.barplot(data=melted, x="Metric Group", y="Value", hue="Model")
    plt.xticks(rotation=45, ha='right')
    plt.title("Model Metric Comparison (Grouped by Target & Metric)")
    plt.tight_layout()
    st.pyplot(plt.gcf())

def show_top_bottom_predictions():
    st.subheader("🔝 Top & Bottom 3 Predictions")

    model_names = {
        "naive": "Naive Model",
        "traditional": "Traditional Model",
        "deep_learning": "Deep Learning Model"
    }

    cols = st.columns(3)

    for idx, model_key in enumerate(model_names.keys()):
        df = load_csv(f"{model_key}_predictions.csv")
        if df is None:
            continue

        df = df.drop_duplicates(subset="Golfer")
        with cols[idx]:
            st.markdown(f"**{model_names[model_key]}**")
            for target in ["Victory", "Made Cut"]:
                col_name = f"{target} Prob"
                if col_name in df.columns:
                    st.markdown(f"**Top 3 – {target}**")
                    st.dataframe(df[["Golfer", col_name]]
                        .sort_values(by=col_name, ascending=False)
                        .head(3)
                        .reset_index(drop=True)
                        .style.format({col_name: "{:.4f}"}))

                    st.markdown(f"**Bottom 3 – {target}**")
                    st.dataframe(df[["Golfer", col_name]]
                        .sort_values(by=col_name, ascending=True)
                        .head(3)
                        .reset_index(drop=True)
                        .style.format({col_name: "{:.4f}"}))

def show_evaluation():
    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        st.subheader("📄 Evaluation Summary Table")
        st.dataframe(df)

        plot_metric_comparison(df)
        show_top_bottom_predictions()
    else:
        st.error("Evaluation summary not found.")

# Streamlit app
st.set_page_config(page_title="Masters Prediction App", layout="wide")
st.title("🏌️ Masters Tournament Prediction Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Naive Model", "Traditional Model", "Deep Learning Model", "Evaluation"])

with tab1:
    show_predictions("Naive Model Predictions", "naive_predictions.csv")

with tab2:
    show_predictions("Traditional Model Predictions", "traditional_predictions.csv")

with tab3:
    show_predictions("Deep Learning Model Predictions", "deep_learning_predictions.csv")

with tab4:
    show_evaluation()

# 🏌️ Masters Tournament Prediction System

This repository contains a complete deep learning pipeline for predicting outcomes of the 2025 Masters golf tournament. Built as a graduate-level project at Duke University, the system supports a publicly deployed interface for exploring model predictions. The project evaluates three modeling approaches—naive, classical machine learning, and deep learning—and includes a reproducible training pipeline and an interactive Streamlit dashboard.

---

## 🔍 Problem Statement

This project tackles the challenge of predicting golf tournament outcomes using a combination of:

- Player bios (unstructured text)
- Official player statistics (OWGR and strokes gained)
- Tournament metadata (course, weather, and preview articles)

We predict:
- 🏆 Probability of winning (Victory)
- 🖐️ Probability of Top 5 finish
- 🔟 Probability of Top 10 finish
- ✂️ Probability of making the cut

I wanted to test if including textual sentiment data from previews and course information could match the performance of more traditional numbers-based ML analytical methods that are commonplace in sports. While not seen traditionally as NLP (chat/text generation) or a Recommendation System, I believe my DL approach qualifies as a hybrid of the two because I am using embeddings of biographical and tournament metadata information and scoring player x tournamnet, much like user x movie or other recommendation system examples. 

---

## 🧠 Modeling Approaches

### 1. Naive Model

- **Approach**: Uses fractional sportsbook odds converted to probabilities.
- **Assumption**: Market odds reflect underlying performance probabilities.
- **Extras**: Top 5 / Top 10 / Cut heuristics based on ranking.
- **Script**: `scripts/naive_model.py`

### 2. Traditional Machine Learning

- **Model**: Random Forest with GridSearchCV
- **Inputs**: OWGR rankings, strokes gained metrics from 10+ tournaments, odds.
- **Targets**: Binary classification for each outcome.
- **Script**: `scripts/traditional_model.py`

### 3. Deep Learning Model

- **Model**: Multilayer Perceptron (PyTorch)
- **Inputs**:
  - SentenceTransformer embeddings of player bios
  - Tournament-level metadata embedding (previews, weather reports, course stats and information)
- **Architecture**
- Concatenated embeddings flow through a 3-layer MLP with ReLU activations and dropout regularization.
- **Training**: Custom `Dataset`, `DataLoader`, MLP architecture
- **Output**: 4 sigmoid-based probabilities
- **Script**: `scripts/deep_learning_model.py`

---

## 🗂 Repository Structure

<pre>
├── app.py                      # Streamlit dashboard UI
├── setup.py                    # Full project setup pipeline
├── requirements.txt            # Required packages
├── .gitignore
├── scripts/
│   ├── naive_model.py
│   ├── traditional_model.py
│   ├── traditional_preprocessing.py
│   ├── deep_learning_model.py
│   ├── embed_bios_and_metadata.py
│   └── evaluate_models.py
├── models/                     # Trained model files (.pkl)
├── data/
│   ├── raw/                    # Raw CSVs (OWGR, tournament results, etc.)
│   ├── processed/              # Embeddings and predictions
│   └── eval/                   # Actual 2025 Masters results + evaluation output
</pre>

---

## 🏃‍♂️ How to Run This Project

### 1. Clone the Repo

```bash
git clone https://github.com/breese5/DL_Final_Project_Masters.git
cd DL_Final_Project_Masters
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv clean_venv
source clean_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Setup Script

```bash
python setup.py
This script handles:

-Embedding bios and metadata
-Running all 3 models
-Storing predictions and evaluations
```

### 4. Launch the App Locally

```bash
streamlit run app.py
```

### 5. Deployment
The Streamlit app is publicly deployed and accessible here (may be shut down if you are viewing the project after release):
🌐 http://34.226.190.15:8501

The system is hosted on AWS EC2, and the .pem key is available upon request if re-deployment or SSH access is required. Reach out!

### 6.Evaluation Metrics
All three models were evaluated using:

- **Brier Score**: Measures the mean squared difference between predicted probabilities and actual binary outcomes, rewarding well-calibrated confidence.
- **Accuracy (Threshold 0.5)**: Percentage of correct predictions based on whether predicted probabilities cross a 0.5 threshold.
- **F1 Score**: Harmonic mean of precision and recall, balancing false positives and false negatives.
- **Recall**: Measures the proportion of actual positives that were correctly identified by the model.

### 7. Model Choice/Future Work

For this project, the traditional model (Random Forest using structured performance metrics) was selected as the most practical approach. While the deep learning model introduced a unique recommendation-style architecture with NLP embeddings and tournament metadata, the limited and relatively shallow textual data likely lacked the depth needed for meaningful sentiment analysis (evidence by the small spread between players in the predicted probabilities).

Additionally, sports outcomes—especially in a single tournament—are inherently noisy and difficult to predict. Evaluation metrics like accuracy, F1, and recall are imperfect in this case due to high class imbalance (e.g., only one winner) and the challenge of setting meaningful thresholds for probabilistic outcomes like top 5 or top 10.

Future work would involve gathering more robust textual data for sentiment analysis and prediction across a season and exploring multi-modal models like player interviews or broadcasts for body language analysis.

### 8. Ethics Statement

This project uses publicly available data only. It does not promote or facilitate gambling, and odds are only used because they have proven to offer strong predictive power. All models are probabilistic and intended for educational purposes. No personal or private data is processed.

### License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with proper attribution.


Bryant Reese
Master of Engineering in AI
Duke University

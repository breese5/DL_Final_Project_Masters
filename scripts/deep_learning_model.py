import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

# ----------------------------
#  Dataset and Model Classes
# ----------------------------

class MastersDataset(Dataset):
    """
    Custom PyTorch dataset for combining player bio embeddings with tournament metadata.
    Each sample is a concatenation of player embedding and metadata.
    """
    def __init__(self, player_embeds, metadata_embed, labels):
        self.player_embeds = player_embeds
        self.metadata_embed = metadata_embed
        self.labels = labels

    def __len__(self):
        return len(self.player_embeds)

    def __getitem__(self, idx):
        player_vec = self.player_embeds[idx]
        combined_vec = np.concatenate([player_vec, self.metadata_embed])
        label_vec = self.labels[idx]
        return torch.tensor(combined_vec, dtype=torch.float32), torch.tensor(label_vec, dtype=torch.float32)


class PredictionMLP(nn.Module):
    """
    A simple MLP model to predict 4 targets from combined embeddings.
    """
    def __init__(self, input_size, hidden_size=128):
        super(PredictionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),  # 4 targets: Victory, Top 5, Top 10, Made Cut
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
#        Training Loop
# ----------------------------

def train(model, dataloader, criterion, optimizer, epochs=20):
    """
    Standard PyTorch training loop.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

# ----------------------------
#         Main Logic
# ----------------------------

def main():
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "processed")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load embeddings
    player_embed_df = pd.read_csv(os.path.join(data_dir, "player_bio_embeddings.csv"))
    metadata_embed = pd.read_csv(os.path.join(data_dir, "tournament_metadata_embedding.csv")).values[0]

    # Load and process labels from traditional model
    labels_df = pd.read_csv(os.path.join(data_dir, "traditional_predictions.csv"))
    player_embed_df["Golfer"] = player_embed_df["Golfer"].str.lower().str.strip()
    labels_df["Golfer"] = labels_df["Golfer"].str.lower().str.strip()

    # Merge by Golfer
    merged = pd.merge(player_embed_df, labels_df, on="Golfer", how="inner")
    if merged.empty:
        raise ValueError("Merged dataset is empty. Check player name formatting.")

    # Separate data
    player_embeddings = merged.drop(columns=["Golfer"] + [c for c in merged.columns if "Prob" in c]).values
    targets = merged[[c for c in merged.columns if "Prob" in c]].values

    # Dataset and DataLoader
    dataset = MastersDataset(player_embeddings, metadata_embed, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model
    input_dim = player_embeddings.shape[1] + metadata_embed.shape[0]
    model = PredictionMLP(input_size=input_dim)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Train
    train(model, dataloader, criterion, optimizer, epochs=25)

    # Save model
    with open(os.path.join(model_dir, "deep_learning_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/deep_learning_model.pkl")

    # Save predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            preds = model(inputs).numpy()
            all_preds.append(preds)
    all_preds = np.vstack(all_preds)

    result_df = pd.DataFrame(all_preds, columns=["Victory Prob", "Top 5 Prob", "Top 10 Prob", "Made Cut Prob"])
    result_df["Golfer"] = merged["Golfer"].values
    result_df = result_df[["Golfer", "Victory Prob", "Top 5 Prob", "Top 10 Prob", "Made Cut Prob"]]
    result_df.to_csv(os.path.join(data_dir, "deep_learning_predictions.csv"), index=False)
    print("✅ Predictions saved to data/processed/deep_learning_predictions.csv")

if __name__ == "__main__":
    main()

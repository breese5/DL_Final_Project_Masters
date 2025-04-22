import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def get_project_paths():
    """
    Sets and returns key paths for the project directory.
    
    Returns:
        tuple: paths for raw data and processed data
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)
    return raw_path, processed_path

def load_data(raw_path):
    """
    Loads bios and metadata CSV files.

    Args:
        raw_path (str): path to the raw data directory

    Returns:
        tuple: DataFrames for bios and metadata
    """
    bios_path = os.path.join(raw_path, "masters_player_bios.csv")
    metadata_path = os.path.join(raw_path, "masters_metadata.csv")
    bios_df = pd.read_csv(bios_path, encoding="ISO-8859-1")
    metadata_df = pd.read_csv(metadata_path, encoding="ISO-8859-1")
    return bios_df, metadata_df

def embed_texts(model, texts, desc="Embedding"):
    """
    Generates sentence embeddings with progress bar.

    Args:
        model (SentenceTransformer): pretrained transformer model
        texts (list or Series): list of texts to encode
        desc (str): tqdm description label

    Returns:
        list: list of embedding vectors
    """
    return [model.encode(text) for text in tqdm(texts, desc=desc)]

def embed_and_save(bios_df, metadata_df, processed_path):
    """
    Embeds bios and tournament metadata, then saves as CSV and NPY.

    Args:
        bios_df (DataFrame): player bios
        metadata_df (DataFrame): tournament metadata
        processed_path (str): path to store outputs
    """
    # Combine bio text
    bios_df["bio_text"] = bios_df["Bio"].fillna("")
    metadata_text = " ".join(metadata_df.iloc[0].dropna().astype(str).tolist())

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed bios
    bio_embeddings = embed_texts(model, bios_df["bio_text"], desc="Embedding bios")
    bio_embed_df = pd.DataFrame(bio_embeddings)
    bio_embed_df["Golfer"] = bios_df["Golfer"]
    bio_embed_df.to_csv(os.path.join(processed_path, "player_bio_embeddings.csv"), index=False)

    # Embed metadata
    metadata_embedding = model.encode(metadata_text)
    meta_embed_df = pd.DataFrame([metadata_embedding])
    meta_embed_df.to_csv(os.path.join(processed_path, "tournament_metadata_embedding.csv"), index=False)

    # Save as .npy
    np.save(os.path.join(processed_path, "player_embeddings.npy"), np.array(bio_embeddings))
    np.save(os.path.join(processed_path, "metadata_embedding.npy"), metadata_embedding)

    print("✅ Embeddings saved to CSV and .npy files.")

def main():
    """
    Main function to execute embedding workflow from CLI.
    """
    raw_path, processed_path = get_project_paths()
    bios_df, metadata_df = load_data(raw_path)
    embed_and_save(bios_df, metadata_df, processed_path)

if __name__ == "__main__":
    main()

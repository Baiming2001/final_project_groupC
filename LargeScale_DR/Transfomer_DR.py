import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import time
import psutil
import os
import gc

# === Set device: CPU aligned with the Megaman Paper ===
device = torch.device("cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # MPS acc
print("Using device:", device)


class TransformerDimReducer(nn.Module):
    """
    A lightweight Transformer encoder for dimensionality reduction.

    Args:
        input_dim (int): Input embedding dimension (default 300).
        hidden_dim (int): Transformer hidden size.
        latent_dim (int): Bottleneck dimension (default 2).
        nhead (int): Number of attention heads.

    Structure:
        - Linear embedding: input_dim → hidden_dim
        - TransformerEncoder: hidden_dim with 1 layer
        - Projector: hidden_dim → latent_dim (2D bottleneck)
        - Decoder: latent_dim → input_dim (reconstruction)
    """
    def __init__(self, input_dim=300, hidden_dim=128, latent_dim=2, nhead=4):
        super(TransformerDimReducer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.projector = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq=1, input_dim)
        x = self.embedding(x)          # (batch, 1, hidden_dim)
        x = x.permute(1, 0, 2)         # (seq=1, batch, hidden_dim) for transformer
        encoded = self.encoder(x)      # (seq=1, batch, hidden_dim)
        encoded = encoded.squeeze(0)   # (batch, hidden_dim)
        latent = self.projector(encoded)  # (batch, latent_dim)
        decoded = self.decoder(latent)    # (batch, input_dim)
        return latent, decoded


def print_memory_usage(stage):
    """
    Prints current memory usage in GB.

    Args:
        stage (str): Description of the current stage.
    """
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"{stage} Memory Usage: {mem_gb:.2f} GB")


def load_vectors_batch(model, start, end):
    """
    Load a batch of word vectors from a Gensim model.

    Args:
        model: Gensim KeyedVectors model.
        start (int): Start index.
        end (int): End index.

    Returns:
        np.ndarray: Array of vectors.
    """
    words = model.index_to_key[start:end]
    vectors = np.stack([model[word] for word in words])
    return vectors


def train_transformer(model_path, batch_size=100000, epochs=5, latent_dim=2):
    """
    Train the transformer-based model for dimensionality reduction.

    Args:
        model_path (str): Path to the Word2Vec binary file.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs.
        latent_dim (int): Target dimension (bottleneck).

    Returns:
        float: Training time in minutes.
        TransformerDimReducer: trained model.
    """
    print("Loading model index...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=None)
    total_words = len(model.index_to_key)
    print(f"Total words in model: {total_words}")

    transformer = TransformerDimReducer(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print_memory_usage("Before training")
    train_start = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        running_loss = 0.0
        batch_count = 0

        for i in range(0, total_words, batch_size):
            end_idx = min(i + batch_size, total_words)
            vectors_np = load_vectors_batch(model, i, end_idx)
            vectors = torch.tensor(vectors_np, dtype=torch.float32, device=device).unsqueeze(1)  # (batch, seq=1, 300)

            optimizer.zero_grad()
            latent, recon = transformer(vectors)
            loss = criterion(recon, vectors.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            print(f" Batch {batch_count}, samples {i}-{end_idx}, loss: {loss.item():.6f}", end='\r')

        avg_loss = running_loss / batch_count
        print(f"\nEpoch {epoch+1} finished, avg loss: {avg_loss:.6f}")
        print_memory_usage(f"After epoch {epoch+1}")

    train_time_min = (time.time() - train_start) / 60
    print(f"Training finished in {train_time_min:.2f} minutes")
    print_memory_usage("After training")

    torch.save(transformer.state_dict(), "transformer_google_news.pth")
    print("Model saved as transformer_google_news.pth")

    transformer.eval()
    with torch.no_grad():
        sample_vectors = torch.tensor(load_vectors_batch(model, 0, 10), dtype=torch.float32, device=device).unsqueeze(1)
        sample_latent, _ = transformer(sample_vectors)
        latent_np = sample_latent.cpu().numpy()
        print("Sample latent vectors:\n", latent_np)
        print("Latent shape:", latent_np.shape)
        print("Latent std per dim:", np.std(latent_np, axis=0))
        print("Latent mean per dim:", np.mean(latent_np, axis=0))

    return train_time_min, transformer


def infer_embeddings(model_path, transformer_path, batch_size=100000, latent_dim=2):
    """
    Run the trained transformer encoder to get low-dimensional embeddings.

    Args:
        model_path (str): Path to Word2Vec model.
        transformer_path (str): Path to saved transformer weights.
        batch_size (int): Batch size for inference.
        latent_dim (int): Bottleneck dimension.

    Returns:
        embeddings (np.ndarray): Compressed embeddings.
        float: Inference time in minutes.
    """
    print("Loading word vectors for inference...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=None)
    total_words = len(model.index_to_key)
    print(f"Total words: {total_words}")

    transformer = TransformerDimReducer(latent_dim=latent_dim).to(device)
    transformer.load_state_dict(torch.load(transformer_path))
    transformer.eval()

    embeddings_list = []
    print_memory_usage("Before inference")
    infer_start = time.time()

    with torch.no_grad():
        for i in range(0, total_words, batch_size):
            end_idx = min(i + batch_size, total_words)
            vectors_np = np.stack([model[word] for word in model.index_to_key[i:end_idx]])
            vectors = torch.tensor(vectors_np, dtype=torch.float32, device=device).unsqueeze(1)

            latent, _ = transformer(vectors)
            embeddings_list.append(latent.cpu().numpy())

            print(f"Inferred batch {i} to {end_idx}", end='\r')
            if (i // batch_size) % 5 == 0:
                print_memory_usage(f"During inference batch {i // batch_size}")

    embeddings = np.vstack(embeddings_list)
    np.save("transformer_latent_2D.npy", embeddings)

    infer_time_min = (time.time() - infer_start) / 60
    print(f"\nInference finished in {infer_time_min:.2f} minutes")
    print_memory_usage("After inference")

    print("Latent vector stats:")
    print(f"Mean per dim: {np.mean(embeddings, axis=0)}")
    print(f"Std per dim: {np.std(embeddings, axis=0)}")

    return embeddings, infer_time_min


def visualize_embeddings(embeddings, model_path, num_words=1000):
    """
    Visualizes the 2D embeddings with word labels.

    Args:
        embeddings (np.ndarray): 2D embeddings.
        model_path (str): Path to original Word2Vec model.
        num_words (int): Number of words to display.
    """
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=num_words)
    words = model.index_to_key
    coords = embeddings[:num_words]

    plt.figure(figsize=(14, 12))
    plt.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.6)

    for i in range(len(words)):
        plt.text(coords[i, 0], coords[i, 1], words[i], fontsize=6, alpha=0.6)

    plt.title("2D Transformer Word Embedding Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Transformer_2D_visualization.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = "GoogleNews-vectors-negative300.bin"
    TRANSFORMER_PATH = "transformer_google_news.pth"

    total_start = time.time()

    # Train
    train_time_min, trained_transformer = train_transformer(MODEL_PATH, batch_size=100000, epochs=5, latent_dim=2)

    # Inference
    embeddings, infer_time_min = infer_embeddings(MODEL_PATH, TRANSFORMER_PATH, batch_size=100000, latent_dim=2)

    # Visualization
    visualize_embeddings(embeddings, MODEL_PATH, num_words=1000)

    # Final Summary
    print("\n✅ Final Summary:")
    print(f"Training time: {train_time_min:.2f} minutes")
    print(f"Inference time: {infer_time_min:.2f} minutes")
    print(f"Total runtime: {(time.time() - total_start) / 60:.2f} minutes")
    print_memory_usage("Final")

    gc.collect()

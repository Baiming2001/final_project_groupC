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


class Autoencoder(nn.Module):
    """
    A simple autoencoder model for dimensionality reduction.

    Args:
        input_dim (int): Dimensionality of input vectors.
        bottleneck_dim (int): Target dimensionality (compressed).

    Structure:
        - Encoder: 300 → 128 → bottleneck_dim
        - Decoder: bottleneck_dim → 128 → 300
    """
    def __init__(self, input_dim=300, bottleneck_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
            nn.Identity()  # Remove nonlinearity to avoid saturation
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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


def train_autoencoder(model_path, batch_size=100000, epochs=5, bottleneck_dim=2):
    """
    Trains an autoencoder to reduce dimensionality of word vectors.

    Args:
        model_path (str): Path to the Word2Vec binary file.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs.
        bottleneck_dim (int): Target dimension.

    Returns:
        float: Training time in minutes.
    """
    print("Loading model index...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=None)
    total_words = len(model.index_to_key)
    print(f"Total words in model: {total_words}")

    autoencoder = Autoencoder(input_dim=300, bottleneck_dim=bottleneck_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
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
            vectors = torch.tensor(vectors_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            outputs = autoencoder(vectors)
            loss = criterion(outputs, vectors)
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

    torch.save(autoencoder.state_dict(), "autoencoder_google_news.pth")
    print("Model saved as autoencoder_google_news.pth")

    autoencoder.eval()
    with torch.no_grad():
        sample_vectors = torch.tensor(load_vectors_batch(model, 0, 10), dtype=torch.float32, device=device)
        encoded_sample = autoencoder.encoder(sample_vectors)
        latent_np = encoded_sample.cpu().numpy()
        print("Sample latent vectors:\n", latent_np)
        print("Latent shape:", latent_np.shape)
        print("Latent std per dim:", np.std(latent_np, axis=0))
        print("Latent mean per dim:", np.mean(latent_np, axis=0))
    return train_time_min


def infer_embeddings(model_path, autoencoder_path, batch_size=100000, bottleneck_dim=2):
    """
    Runs the trained encoder to obtain low-dimensional embeddings.

    Args:
        model_path (str): Path to Word2Vec model.
        autoencoder_path (str): Path to saved autoencoder weights.
        batch_size (int): Batch size during inference.
        bottleneck_dim (int): Output dimension.

    Returns:
        embeddings (np.ndarray): Compressed embeddings.
        float: Inference time in minutes.
    """
    print("Loading word vectors for inference...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=None)
    total_words = len(model.index_to_key)
    print(f"Total words: {total_words}")

    autoencoder = Autoencoder(input_dim=300, bottleneck_dim=bottleneck_dim).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.eval()

    embeddings_list = []
    print_memory_usage("Before inference")
    infer_start = time.time()

    with torch.no_grad():
        for i in range(0, total_words, batch_size):
            end_idx = min(i + batch_size, total_words)
            vectors_np = np.stack([model[word] for word in model.index_to_key[i:end_idx]])
            vectors = torch.tensor(vectors_np, dtype=torch.float32, device=device)

            encoded = autoencoder.encoder(vectors)
            embeddings_list.append(encoded.cpu().numpy())

            print(f"Inferred batch {i} to {end_idx}", end='\r')
            if (i // batch_size) % 5 == 0:
                print_memory_usage(f"During inference batch {i // batch_size}")

    embeddings = np.vstack(embeddings_list)
    np.save("word2vec_latent_2D.npy", embeddings)

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

    plt.title("2D Autoencoder Word Embedding Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Autoencoder_2D_visualization.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = "GoogleNews-vectors-negative300.bin"
    AUTOENCODER_PATH = "autoencoder_google_news.pth"

    total_start = time.time()

    # Train
    train_time_min = train_autoencoder(MODEL_PATH, batch_size=100000, epochs=5, bottleneck_dim=2)

    # Inference
    embeddings, infer_time_min = infer_embeddings(MODEL_PATH, AUTOENCODER_PATH, batch_size=100000, bottleneck_dim=2)

    # Visualization
    visualize_embeddings(embeddings, MODEL_PATH, num_words=1000)

    # Final Summary
    print("\n✅ Final Summary:")
    print(f"Training time: {train_time_min:.2f} minutes")
    print(f"Inference time: {infer_time_min:.2f} minutes")
    print(f"Total runtime: {(time.time() - total_start) / 60:.2f} minutes")
    print_memory_usage("Final")

    gc.collect()

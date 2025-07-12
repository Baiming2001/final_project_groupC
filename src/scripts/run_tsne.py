import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.dimensionality_utils import *


if __name__ == "__main__":
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # 调用有些问题 bin_path = os.path.join(base_dir, "..", "..", "data", "word2vec", "text8_vectors.bin")
    bin_path = os.path.join(project_root, "data", "word2vec", "text8_vectors.bin")
    fig_path = os.path.join(project_root, "data", "word2vec", "tsne_embedding.png")
    # Load vectors
    vectors, labels = load_word_vectors(bin_path, top_n=10000)

    # Run t-SNE Embedding
    X_2d = run_tsne(vectors, n_components=2, perplexity=30, init='pca', random_state=42)

    # Plot & Save
    plot_2d_embedding(X_2d, labels, save_path=fig_path)

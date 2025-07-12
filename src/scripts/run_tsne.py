import os
import sys

# 获取当前文件目录的上级目录（src）
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.dimensionality_utils import *
if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(base_dir, "..", "data", "word2vec", "text8_vectors.bin")
    fig_path = os.path.join(base_dir, "..", "data", "word2vec", "tsne_embedding.png")

    # Load vectors
    vectors, labels = load_word_vectors(bin_path, top_n=10000)

    # Run t-SNE Embedding
    X_2d = run_tsne(vectors, n_components=2, perplexity=30, init='pca', random_state=42)

    # Plot & Save
    plot_2d_embedding(X_2d, labels, save_path=fig_path)



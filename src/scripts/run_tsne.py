import os
import sys
import numpy as np
from sklearn.datasets import make_swiss_roll

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.dimensionality_utils import *

if __name__ == "__main__": 
    def tsne_on_dataset(dataset, n_samples, n_extra_dims):
        # choose dataset and use diffusion map to reduce dimensionality to 2
        if dataset == "swissroll": 
            # generate swiss roll dataset in 3 dimensions without noise
            X_3d, _ = make_swiss_roll(n_samples=n_samples, noise=0.0)

            # add extra Gaussian noise dimensions to the original swissroll dataset
            noise_dims = np.random.randn(X_3d.shape[0], n_extra_dims)
            X_extra_dims = np.hstack((X_3d, noise_dims)) 

            # use t-SNE on swiss roll dataset to reduce the dimensionality to 2
            X_2d = run_tsne(X_extra_dims, n_components = 2, perplexity=30, init='pca', random_state=42)

            # plot and save
            fig_path = os.path.join(project_root, "data", "swissroll", "tsne_embedding.png")
            plot_2d_embedding(X_2d, title = f"t-SNE Embedding of Swiss Roll (n={n_samples}, d={n_extra_dims + 3})", save_path=fig_path)


        elif dataset == "word2vec": 
            # load certain number (n_samples) of most frequent vectors in word2vec
            bin_path = os.path.join(project_root, "data", "word2vec", "text8_vectors_tsne.bin")
            vectors, _ = load_word_vectors(bin_path, top_n=n_samples)

            # use t-SNE on word2vec dataset to reduce the dimensionality to 2
            X_2d = run_tsne(vectors, n_components = 2, perplexity=30, init='pca', random_state=42)

            # plot and save
            fig_path = os.path.join(project_root, "data", "word2vec", "tsne_embedding.png")
            plot_2d_embedding(X_2d, title = f"t-SNE Embedding of Word2Vec (top {n_samples} words)", save_path=fig_path)

        else: 
            raise ValueError(f"Unknown dataset '{dataset}'. Please choose either 'swissroll' or 'word2vec'.")

    # main
    # swissroll
    n_samples_swissroll = 10000
    n_extra_dims = 0
    tsne_on_dataset("swissroll", n_samples=n_samples_swissroll, n_extra_dims=n_extra_dims)

    # word2vec
    n_samples_word2vec = 10000
    tsne_on_dataset("word2vec", n_samples=n_samples_word2vec, n_extra_dims=None)


            

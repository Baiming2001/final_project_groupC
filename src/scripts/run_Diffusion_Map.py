import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.dimensionality_utils import *
from src.swissroll import swiss_roll_dataset

if __name__ == "__main__":
    
    ############################################
    # Swiss Roll Dataset
    ############################################

    # paths for swiss roll
    fig_path_swissroll = os.path.join(project_root, "data", "swissroll", "diffusion_maps.png")

    # generate swiss roll dataset
    datapoints=10000
    length_phi=0.004*datapoints*np.pi
    length_Z=0.001*datapoints
    phi, _, vectors = swiss_roll_dataset(length_phi = length_phi, length_Z = length_Z, n_dimensions=20, datapoints=datapoints)

    # run Diffusion Maps on swiss roll dataset
    X_2d = run_diffusion_maps(vectors, n_components=3, epsilon=0.5)
    plot_2d_embedding_colored(X_2d[:, 1:3], phi, title="Dimensionality Reduction with epsilon=0.5, colored with phi", save_path=fig_path_swissroll)
    
    ############################################
    # Word2Vec Dataset
    ############################################

    # Load vectors
    # for word2vec
    # bin_path = os.path.join(project_root, "data", "word2vec", "text8_vectors.bin")
    # vectors, labels = load_word_vectors(bin_path, top_n=1000)


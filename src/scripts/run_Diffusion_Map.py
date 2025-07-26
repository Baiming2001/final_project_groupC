import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.dimensionality_utils import *

if __name__ == "__main__":
    
    ############################################
    # Swiss Roll Dataset
    ############################################

    def diffusion_map_for_swissroll(n_samples, n_extra_dims, n_components): 
        """
        Apply Diffusion Maps with automatic eigenvector selection on a synthetic Swiss Roll dataset.

        This function generates a 3D Swiss Roll dataset, optionally augments it with 
        additional Gaussian noise dimensions, and performs nonlinear dimensionality 
        reduction using Diffusion Maps. The most informative eigenvectors are selected 
        using `LocalRegressionSelection`, and the resulting 2D embedding is visualized 
        and saved as a scatter plot colored by the data's intrinsic structure.

        Parameters
        ----------
        n_samples : int
            Number of data points to generate in the Swiss Roll dataset.

        n_extra_dims : int
            Number of additional Gaussian noise dimensions to append to the data,
            increasing the ambient dimensionality.

        n_components : int
            Number of diffusion components (eigenvectors) to compute using Diffusion Maps.

        Returns
        -------
        None
            The function produces and saves a 2D scatter plot to the local file system.
            No data is returned.
        """
        # generate swiss roll dataset in 3 dimensions without noise
        X_3d, X_color = make_swiss_roll(n_samples=n_samples, noise=0.0)

        # add extra Gaussian noise dimensions to the original swissroll dataset
        noise_dims = np.random.randn(X_3d.shape[0], n_extra_dims)
        X_extra_dims = np.hstack((X_3d, noise_dims)) 

        # use Diffusion Maps on swiss roll dataset to reduce the dimensionality to 2
        dmap = run_diffusion_maps(X_extra_dims, n_components = n_components)

        # find best eigenvector using LocalRegressionSelection
        selection = LocalRegressionSelection(
            intrinsic_dim=2, 
            n_subsample=min(1000, n_samples * 0.2), 
            strategy="dim"
            ).fit(dmap.eigenvectors_)

        # paths for saving the plot  result 
        fig_filename = f"diffusion_maps_n{n_samples}_d{n_extra_dims + 3}.png"
        fig_path_swissroll = os.path.join(project_root, "data", "swissroll", fig_filename)

        # reduce number of points for plotting
        rng = np.random.default_rng(1)
        nr_samples_plot = int(n_samples * 0.1)
        idx_plot = rng.permutation(n_samples)[0:nr_samples_plot]

        # plot the selection result
        target_mapping = selection.transform(dmap.eigenvectors_)
        f, ax = plt.subplots(figsize=(15, 9))
        ax.scatter(
            target_mapping[idx_plot, 0],
            target_mapping[idx_plot, 1],
            c=X_color[idx_plot],
            cmap=plt.cm.Spectral,
        )

        # add title to the plot
        evec_indices = selection.evec_indices_
        ax.set_title(rf"Eigenvectors selected by the diffusion map method: $\Psi_1$ and $\Psi_{{{evec_indices[1]}}}$", fontsize=16)

        # add label for x- and y-axis, save to fig_path_swissroll
        ax.set_xlabel(rf"$\Psi_1$", fontsize=12)
        ax.set_ylabel(rf"$\Psi_{{{evec_indices[1]}}}$", fontsize=12)
        plt.savefig(fig_path_swissroll)

        # optional: plot pairing eigenvectors, ignore first eigenvector phi_0
        # plot_pairwise_eigenvector(
        #     eigenvectors=dmap.eigenvectors_[idx_plot, :], 
        #     n=1,
        #     fig_params=dict(figsize=[15, 15]), 
        #     scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot])
        # )

        # show plot result
        plt.show()

    # main
    n_samples = 10000
    n_extra_dims = 0
    n_components = 10
    diffusion_map_for_swissroll(n_samples=n_samples, n_extra_dims=n_extra_dims, n_components=n_components)

    ############################################
    # Word2Vec Dataset
    ############################################




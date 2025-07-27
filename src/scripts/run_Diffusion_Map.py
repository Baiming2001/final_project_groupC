import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_swiss_roll
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.dimensionality_utils import *

if __name__ == "__main__": 

    def diffusion_map_on_dataset(dataset, n_samples, n_extra_dims, n_components): 
        """
        Apply Diffusion Maps with automatic eigenvector selection on a given dataset.

        This function performs nonlinear dimensionality reduction using Diffusion Maps
        on either a synthetic Swiss Roll dataset or a pre-trained Word2Vec embedding.
        The most informative diffusion components are selected using`LocalRegressionSelection`, 
        and the resulting 2D embedding is visualized and saved as a scatter plot.

        Parameters
        ----------
        dataset : str
            Name of the dataset to use. Must be one of:
            - "swissroll": generates a 3D Swiss Roll dataset with optional Gaussian noise dimensions.
            - "word2vec": loads a Word2Vec embedding from file.

        n_samples : int
            Number of data points to use:
            - For "swissroll", this is the number of points generated.
            - For "word2vec", this controls how many top vectors are loaded.

        n_extra_dims : int or None
            Only used for dataset "swissroll". Number of additional Gaussian noise dimensions
            to append to the 3D Swiss Roll. Ignored for "word2vec".

        n_components : int
            Number of eigenvectors (diffusion components) to compute during
            Diffusion Maps embedding.

        Returns
        -------
        None
            The function produces and saves a 2D scatter plot showing the data
            embedded via the selected diffusion components. No values are returned.
        """
        # choose dataset and use diffusion map to reduce dimensionality to 2
        if dataset == "swissroll": 
            # generate swiss roll dataset in 3 dimensions without noise
            X_3d, X_color = make_swiss_roll(n_samples=n_samples, noise=0.0)

            # add extra Gaussian noise dimensions to the original swissroll dataset
            noise_dims = np.random.randn(X_3d.shape[0], n_extra_dims)
            X_extra_dims = np.hstack((X_3d, noise_dims)) 

            # use Diffusion Maps on swiss roll dataset to reduce the dimensionality to 2
            dmap = run_diffusion_maps(X_extra_dims, n_components = n_components)

        elif dataset == "word2vec": 
            # load certain number (n_samples) of most frequent vectors in word2vec
            bin_path = os.path.join(project_root, "data", "word2vec", "text8_vectors_train.bin")
            vectors, _ = load_word_vectors(bin_path, top_n=n_samples)
            X_color = np.random.rand(vectors.shape[0])

            # use Diffusion Maps on word2vec dataset to reduce the dimensionality to 2
            dmap = run_diffusion_maps(vectors, n_components = n_components)

        else: 
            raise ValueError(f"Unknown dataset '{dataset}'. Please choose either 'swissroll' or 'word2vec'.")
        
        # find best eigenvector using LocalRegressionSelection
        selection = LocalRegressionSelection(
            intrinsic_dim=2, 
            n_subsample=int(n_samples * 0.2), 
            strategy="dim"
            ).fit(dmap.eigenvectors_)

        # reduce number of points for plotting
        rng = np.random.default_rng(1)
        nr_samples_plot = int(n_samples * 0.2)
        idx_plot = rng.permutation(n_samples)[0:nr_samples_plot]

        # plot the selection result
        target_mapping = selection.transform(dmap.eigenvectors_)
        _, ax = plt.subplots(figsize=(15, 9))
        ax.scatter(
            target_mapping[idx_plot, 0],
            target_mapping[idx_plot, 1],
            c=X_color[idx_plot],
            cmap=plt.cm.Spectral,
        )

        # add title to the plot
        evec_indices = selection.evec_indices_
        ax.set_title(rf"Eigenvectors selected by the diffusion map method on {dataset}: $\Psi_1$ and $\Psi_{{{evec_indices[1]}}}$", fontsize=16)

        # add label for x- and y-axis, save to fig_path_swissroll
        ax.set_xlabel(rf"$\Psi_1$", fontsize=12)
        ax.set_ylabel(rf"$\Psi_{{{evec_indices[1]}}}$", fontsize=12)

        # paths for saving the plot result 
        if dataset == "swissroll":
            fig_filename = f"diffusion_maps_swissroll_n{n_samples}_d{n_extra_dims + 3}.png"
            fig_path = os.path.join(project_root, "data", "swissroll", fig_filename)

        elif dataset == "word2vec": 
            fig_filename = f"diffusion_maps_word2vec.png"
            fig_path = os.path.join(project_root, "data", "word2vec", fig_filename)
        plt.savefig(fig_path)

        # optional: plot pairing eigenvectors, ignore first eigenvector phi_0
        # plot_pairwise_eigenvector(
        #     eigenvectors=dmap.eigenvectors_[idx_plot, :], 
        #     n=1,
        #     fig_params=dict(figsize=[15, 15]), 
        #     scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot])
        # )

        # show plot result
        plt.show()
    
    def benchmark_diffusion_map_runtime(mode): 
        # choose mode to do the following experiments and benchmark runtime: 
        # experiment 1: fix D = 100, vary sample size N
        # experiment 2: fix N = 50000, vary data dimension D
        runtimes = []
        x_values = []
        skipped = []
        n_components = 10
        if mode == "vary_N": 
            n_extra_dims = 97
            sample_sizes = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6]
            pbar = tqdm(sample_sizes, desc="Varying N")

            for N in pbar: 
                # estimate memory usage
                est_mem = estimate_memory_GB(N=N)
                if est_mem > 10:
                    print(f"Skipping N={int(N)}: estimated memory {est_mem:.2f} GB > {10} GB")
                    skipped.append(N)
                    continue

                # generate swissroll with 100 dimensions and N samples
                X_3d, _ = make_swiss_roll(n_samples=int(N), noise=0.0)
                noise_dims = np.random.randn(X_3d.shape[0], n_extra_dims)
                X_extra_dims = np.hstack((X_3d, noise_dims)) 

                # compute run time
                start = time.time()
                try:
                    _ = run_diffusion_maps(X_extra_dims, n_components=n_components)
                except MemoryError:
                    print(f"MemoryError at N={int(N)}, skipping...")
                    skipped.append(N)
                    continue
                except Exception as e:
                    print(f"Exception at N={int(N)}: {type(e).__name__} - {e}")
                    skipped.append(N)
                    continue

                end = time.time()
                runtime = end - start

                # record run time and correspondent number of samples N
                pbar.set_postfix({"N": int(N), "runtime": round(runtime, 2)}) 
                runtimes.append(runtime)
                x_values.append(N)

        elif mode == "vary_D": 
            n_samples = 50000
            n_extra_dims = [7, 17, 47, 97, 197, 497, 997, 1997, 4997, 9997]
            pbar = tqdm(n_extra_dims, desc="Varying D")

            for D in pbar: 
                # estimate memory usage
                est_mem = estimate_memory_GB(D=D)
                if est_mem > 10:
                    print(f"Skipping N={int(N)}: estimated memory {est_mem:.2f} GB > {10} GB")
                    skipped.append(N)
                    continue
                # generate swissroll with (D+3) dimensions and 50000 samples
                X_3d, _ = make_swiss_roll(n_samples=n_samples, noise=0.0)
                noise_dims = np.random.randn(X_3d.shape[0], D)
                X_extra_dims = np.hstack((X_3d, noise_dims)) 
            
                # compute run time
                start = time.time()
                try:
                    _ = run_diffusion_maps(X_extra_dims, n_components=n_components)
                except MemoryError:
                    print(f"MemoryError at D={int(D)}, skipping...")
                    skipped.append(D)
                    continue
                except Exception as e:
                    print(f"Exception at D={int(D)}: {type(e).__name__} - {e}")
                    skipped.append(D)
                    continue
                end = time.time()
                runtime = end - start

                # record run time and correspondent number of samples N
                pbar.set_postfix({"D": int(D), "runtime": round(runtime, 2)})
                runtimes.append(runtime)
                x_values.append(N)

        else: 
            raise ValueError(f"Unknown mode '{mode}'. Please choose either 'vary_N' or 'vary_D'.")            

        # plotting
        plt.figure(figsize=(8, 6))
        plt.loglog(x_values, runtimes, marker='o', label='Diffusion Maps (Datafold)')
        xlabel = "Number of samples (N)" if mode == "vary_N" else "Number of dimensions (D)"
        plt.xlabel(xlabel)
        plt.ylabel("Run time (s)")
        plt.title(f"Runtime vs. {xlabel}, 100 features" if mode == "vary_N" else f"Runtime vs. {xlabel}, 50000 samples")
        plt.grid(True)
        if mode == "vary_N":
            plt.xlim(left=1e4)
        elif mode == "vary_D":
            plt.xlim(left=1e1)
        plt.legend()
        plt.tight_layout()

        # save figure
        fig_name = f"runtime_vs_{'N' if mode == 'vary_N' else 'D'}.png"
        fig_path = os.path.join(project_root, "data", "swissroll", fig_name)
        plt.savefig(fig_path)
        plt.show()



    #################################################
    # main
    #################################################
    # swissroll
    n_components_swissroll = 20
    n_samples_swissroll = 10000
    n_extra_dims_swissroll = 97
    # diffusion_map_on_dataset(dataset="swissroll", n_samples=n_samples_swissroll, n_extra_dims=n_extra_dims_swissroll, n_components=n_components_swissroll)
    benchmark_diffusion_map_runtime(mode="vary_N")
    # benchmark_diffusion_map_runtime(mode="vary_D")


    # word2vec
    n_components_wordvec = 20
    n_samples_word2vec = 5000
    # diffusion_map_on_dataset(dataset="word2vec", n_samples=n_samples_word2vec, n_extra_dims=None, n_components=n_components_wordvec)




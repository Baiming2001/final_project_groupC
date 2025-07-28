import os
import sys
import time
import numpy as np
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.utils.dimensionality_utils import *

if __name__ == "__main__": 
    def tsne_on_dataset(dataset, n_samples, n_extra_dims):
        """
        Apply t-SNE on a selected dataset and generate a 2D embedding plot.

        This function supports two datasets:
        - "swissroll": generates a synthetic Swiss Roll dataset with optional 
        Gaussian noise dimensions added to increase ambient dimensionality.
        - "word2vec": loads a Word2Vec embedding and selects the top `n_samples` 
        most frequent word vectors.

        For both datasets, t-SNE is applied to reduce the data to 2 dimensions.
        The resulting embedding is visualized and saved as a PNG figure.

        Parameters
        ----------
        dataset : str
            The dataset to use. Must be one of:
            - "swissroll"
            - "word2vec"

        n_samples : int
            Number of samples to generate or load.
            - For "swissroll", determines the number of points in the generated dataset.
            - For "word2vec", determines the number of most frequent word vectors to load.

        n_extra_dims : int
            Only used for "swissroll". Number of additional Gaussian noise dimensions
            to add to the original 3D Swiss Roll. Ignored for "word2vec".

        Returns
        -------
        None
            The function saves a 2D scatter plot of the t-SNE embedding to the local file system.
            No values are returned.
        """
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
        
    def benchmark_tsne_runtime(mode): 
        """
        Benchmark the runtime of t-SNE under varying dataset configurations.

        This function evaluates the scalability of t-SNE by measuring its runtime 
        on synthetic Swiss roll datasets under two settings:

        1. Varying the number of samples (N) while keeping the dimensionality fixed at 100.
        2. Varying the dimensionality (D) while keeping the number of samples fixed at 50,000.

        For each configuration, a Swiss roll dataset is generated with the desired shape, 
        optionally augmented with Gaussian noise dimensions. The t-SNE embedding is 
        then computed, and the runtime is measured and recorded.

        Memory usage is estimated for each configuration, and experiments that are expected 
        to exceed 10 GB of memory are skipped. Runtime results are visualized on a log-log 
        plot and saved to disk.

        Parameters
        ----------
        mode : str
            Mode of the benchmark. Must be one of:
            - "vary_N": Vary the number of samples while fixing dimensionality (D=100).
            - "vary_D": Vary the dimensionality while fixing the number of samples (N=50000).

        Returns
        -------
        None
            The function saves the runtime plot as a PNG file and displays it.
            No value is returned.
        """
        runtimes = []
        x_values = []
        n_components = 10
        if mode == "vary_N": 
            n_extra_dims = 97
            sample_sizes = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
            pbar = tqdm(sample_sizes, desc="Varying N")

            for N in pbar: 
                # estimate memory usage
                est_mem = tsne_estimate_memory_GB(N=N)
                if est_mem > 10:
                    print(f"Skipping N={int(N)}: estimated memory {est_mem:.2f} GB > {10} GB")
                    continue

                # generate swissroll with 100 dimensions and N samples
                X_3d, _ = make_swiss_roll(n_samples=int(N), noise=0.0)
                noise_dims = np.random.randn(X_3d.shape[0], n_extra_dims)
                X_extra_dims = np.hstack((X_3d, noise_dims)) 

                # compute run time
                start = time.time()
                try:
                    _ = run_tsne(X_extra_dims, n_components = 2, perplexity=30, init='pca', random_state=42)
                except MemoryError:
                    print(f"MemoryError at N={int(N)}, skipping...")
                    continue
                except Exception as e:
                    print(f"Exception at N={int(N)}: {type(e).__name__} - {e}")
                    continue

                end = time.time()
                runtime = end - start

                # record run time and correspondent number of samples N
                pbar.set_postfix({"N": int(N), "runtime": round(runtime, 2)}) 
                runtimes.append(runtime)
                x_values.append(N)

                # early stop to save some time
                if runtime >= 1e4:
                    print(f"Runtime exceeded 1e4 seconds at N={int(N)}, stopping benchmark early.")
                    break

        elif mode == "vary_D": 
            n_samples = 50000
            n_extra_dims = [7, 17, 47, 97, 197, 497, 997, 1997, 4997, 9997]
            pbar = tqdm(n_extra_dims, desc="Varying D")

            for D in pbar: 
                # estimate memory usage
                est_mem = tsne_estimate_memory_GB(D=D+3)
                if est_mem > 10:
                    print(f"Skipping D={int(D+3)}: estimated memory {est_mem:.2f} GB > {10} GB")
                    continue
                # generate swissroll with (D+3) dimensions and 50000 samples
                X_3d, _ = make_swiss_roll(n_samples=n_samples, noise=0.0)
                noise_dims = np.random.randn(X_3d.shape[0], D)
                X_extra_dims = np.hstack((X_3d, noise_dims)) 
            
                # compute run time
                start = time.time()
                try:
                    _ = run_tsne(X_extra_dims, n_components = 2, perplexity=30, init='pca', random_state=42)
                except MemoryError:
                    print(f"MemoryError at D={int(D)}, skipping...")
                    continue
                except Exception as e:
                    print(f"Exception at D={int(D)}: {type(e).__name__} - {e}")
                    continue
                end = time.time()
                runtime = end - start

                # record run time and correspondent number of samples N
                pbar.set_postfix({"D": int(D), "runtime": round(runtime, 2)})
                runtimes.append(runtime)
                x_values.append(D)

                # early stop to save some time
                if runtime >= 1e4:
                    print(f"Runtime exceeded 1e4 seconds at D={int(D)}, stopping benchmark early.")
                    break

        else: 
            raise ValueError(f"Unknown mode '{mode}'. Please choose either 'vary_N' or 'vary_D'.")            

        # plotting
        plt.figure(figsize=(8, 6))
        plt.loglog(x_values, runtimes, marker='o', label='t-SNE')
        xlabel = "Number of samples (N)" if mode == "vary_N" else "Number of dimensions (D)"
        plt.xlabel(xlabel)
        plt.ylabel("Run time (s)")
        plt.title(f"Runtime vs. {xlabel}, 100 features" if mode == "vary_N" else f"Runtime vs. {xlabel}, 50000 samples")
        plt.grid(True)
        if mode == "vary_N":
            plt.xlim(left=1e4, right = 1e6)
            plt.ylim(bottom= 1e0, top = 1e4)
        elif mode == "vary_D":
            plt.xlim(left=1e1, right = 1e4)
            plt.ylim(bottom= 1e0, top = 1e4)
        plt.legend()
        plt.tight_layout()

        # save figure
        fig_name = f"tsne_runtime_vs_{'N' if mode == 'vary_N' else 'D'}.png"
        fig_path = os.path.join(project_root, "data", "swissroll", fig_name)
        plt.savefig(fig_path)
        plt.show()



    #################################################
    # main
    #################################################
    # swissroll
    # swissroll dimensionality reduction visualization
    n_samples_swissroll = 10000
    n_extra_dims = 97
    # tsne_on_dataset("swissroll", n_samples=n_samples_swissroll, n_extra_dims=n_extra_dims)
    
    # experiment varying N and D
    benchmark_tsne_runtime(mode="vary_N")
    benchmark_tsne_runtime(mode="vary_D")

    # word2vec
    # word2vec dimensionality reduction visualization
    n_samples_word2vec = 10000
    # tsne_on_dataset("word2vec", n_samples=n_samples_word2vec, n_extra_dims=None)


            

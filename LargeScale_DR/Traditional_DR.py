"""
Dimensionality Reduction Benchmark on Word2Vec Embeddings

Overview:
This script benchmarks traditional dimensionality reduction techniques 
on pre-trained Word2Vec vectors of varying sizes. 

Motivation:
- For small datasets, classical methods like t-SNE and Spectral Embedding 
  can be effective but still consume significant time and memory.
- As dataset size increases, these traditional methods become prohibitively slow 
  and memory-intensive.
"""

import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csgraph
from gensim.models import KeyedVectors
from memory_profiler import memory_usage
import psutil
import gc

MODEL_PATH = "GoogleNews-vectors-negative300.bin"
BATCH_SIZES = [10000, 50000] # 10000
OUTPUT_CSV = "dim_reduction_time_total_all_methods.csv"
MEMORY_LIMIT_GB = 4  # Reserve memory for system to prevent crash


# def pca_breakdown(X, n_components=2):
#     start = time.time()
#     X_centered = X - X.mean(axis=0)
#     pca = PCA(n_components=n_components)
#     pca.fit(X_centered)
#     embedding = pca.transform(X_centered)
#     return embedding, time.time() - start


def tsne_breakdown(X, n_components=2, perplexity=30):
    start = time.time()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(X)
    print(f"[DEBUG] Running t-SNE on shape: {X.shape}")
    return embedding, time.time() - start


# def isomap_breakdown(X, n_neighbors=15, n_components=2):
#     start = time.time()
#     isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
#     embedding = isomap.fit_transform(X)
#     return embedding, time.time() - start

"""
def spectral_breakdown(X, n_neighbors=20, n_components=2):
    start = time.time()
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nn.kneighbors(X)

    n_samples = X.shape[0]
    row_idx = np.repeat(np.arange(n_samples), n_neighbors)
    col_idx = indices.flatten()
    data = np.ones(len(row_idx))
    adjacency = csr_matrix((data, (row_idx, col_idx)), shape=(n_samples, n_samples))
    adjacency = 0.5 * (adjacency + adjacency.T)

    laplacian, dd = csgraph.laplacian(adjacency, normed=True, return_diag=True)

    embedding_model = SpectralEmbedding(n_components=n_components, affinity='precomputed', random_state=42)
    embedding = embedding_model.fit_transform(adjacency)

    return embedding, time.time() - start
"""

def run_method(name, vectors):
    # if name == "PCA":
    #     return pca_breakdown(vectors)
    if name == "t-SNE":
        return tsne_breakdown(vectors)
    # elif name == "Isomap":
    #     return isomap_breakdown(vectors)
    # elif name == "Spectral":
    #    return spectral_breakdown(vectors)
    else:
        raise ValueError(f"Unsupported method {name}")


def measure_peak_memory(name, vectors):
    timings_container = {}

    def target():
        embedding, total_time = run_method(name, vectors)
        timings_container['total'] = total_time

    mem_usage = memory_usage(target, interval=0.1, timeout=None, max_usage=False)
    peak_mem = max(mem_usage)
    return timings_container, peak_mem


def get_available_memory_gb():
    return psutil.virtual_memory().available / (1024 ** 3)


def main():
    print("Loading word vectors model...")
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
    print("Model loaded!")

    all_results = []

    for batch_size in BATCH_SIZES:
        available_mem = get_available_memory_gb()
        print(f"\nTrying batch_size = {batch_size}, Available memory: {available_mem:.2f} GB")

        # Estimate memory requirement (float32 4 bytes)
        est_mem = batch_size * model.vector_size * 4 / (1024 ** 3)
        print(f"Estimated memory needed: {est_mem:.2f} GB")

        if est_mem > available_mem - MEMORY_LIMIT_GB:
            print(f"Not enough memory, skipping batch_size={batch_size}")
            continue

        try:
            words = model.index_to_key[:batch_size]
            vectors = np.stack([model[word] for word in words])
        except Exception as e:
            print(f"Failed to load word vectors: {e}")
            continue

        for method_name in ["t-SNE", "Spectral"]:
            print(f"Starting method {method_name}...", end=' ', flush=True)
            try:
                timings, peak_mem = measure_peak_memory(method_name, vectors)
                if 'total' not in timings:
                    raise RuntimeError("total timing missing")

                total_min = timings['total'] / 60
                print(f"Done! Total time (minutes): {total_min:.3f} Peak memory (MB): {peak_mem:.2f}")

                all_results.append({
                    "method": method_name,
                    "top_n": batch_size,
                    "memory_peak_mb": round(peak_mem, 3),
                    "time_total_min": round(total_min, 4),
                })

            except Exception as e:
                print(f"Failed: {e}")
                all_results.append({
                    "method": method_name,
                    "top_n": batch_size,
                    "memory_peak_mb": None,
                    "time_total_min": None,
                })

        del vectors
        gc.collect()

    df = pd.DataFrame(all_results)
    print("\nAll test results:")
    print(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

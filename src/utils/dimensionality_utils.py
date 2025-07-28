from sklearn.manifold import SpectralEmbedding, Isomap, TSNE
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

# ========================================
# Dimensionality Reduction Methods
# ========================================

def run_spectral_embedding(vectors, n_components=2, n_neighbors=20):
    """
    Perform dimensionality reduction using Spectral Embedding (Laplacian Eigenmaps).

    This function applies scikit-learn's SpectralEmbedding algorithm to reduce 
    the dimensionality of high-dimensional input data by constructing a neighborhood 
    graph and computing the eigenvectors of the graph Laplacian.

    Parameters
    ----------
    vectors : ndarray of shape (n_samples, n_features)
        The input high-dimensional data to be embedded.
    
    n_components : int, optional (default=2)
        The number of dimensions to project the data onto.

    n_neighbors : int, optional (default=20)
        The number of nearest neighbors used to construct the neighborhood graph.

    Returns
    -------
    X_embedded : ndarray of shape (n_samples, n_components)
        The embedded coordinates of the input data in the lower-dimensional space.
    """
    model = SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity='nearest_neighbors'
    )
    return model.fit_transform(vectors)


def run_tsne(vectors, n_components=2, perplexity=30, init='pca', random_state=42):
    """
    Perform t-SNE dimensionality reduction using scikit-learn.

    This function reduces high-dimensional input vectors to a low-dimensional
    representation using t-distributed Stochastic Neighbor Embedding (t-SNE).
    It is especially useful for visualizing the structure of word embeddings,
    image features, or other high-dimensional data.

    Parameters
    ----------
    vectors : ndarray of shape (n_samples, n_features)
        High-dimensional input data to be embedded.

    n_components : int, optional (default=2)
        Dimension of the embedded space. Typically 2 or 3 for visualization.

    perplexity : float, optional (default=30)
        The perplexity controls the balance between local and global aspects
        of the data. Recommended range is between 5 and 50.

    init : {'pca', 'random'}, optional (default='pca')
        Initialization method for the embedding:
        - 'pca': use PCA projection as initialization (more stable)
        - 'random': use random initialization

    random_state : int, optional (default=42)
        Random seed for reproducibility of the embedding.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        Low-dimensional t-SNE embedding of the input data.
    """
    model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init=init,
        random_state=random_state
    )
    return model.fit_transform(vectors)


def run_isomap(vectors, n_components=2, n_neighbors=10):
    """
    Isomap using scikit-learn.
    """
    model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    return model.fit_transform(vectors)


def run_diffusion_maps(vectors, n_components=10):
    """
    Apply Diffusion Maps for nonlinear dimensionality reduction.

    Diffusion Maps is a spectral embedding technique that models a diffusion 
    process over the dataset, revealing its intrinsic geometric structure. 
    By computing the leading eigenvectors of a diffusion operator, it provides 
    low-dimensional embeddings that preserve the manifold topology.

    This function uses the `datafold` library, which wraps kernel-based manifold 
    learning with data-driven parameter tuning.

    Parameters
    ----------
    vectors : ndarray of shape (n_samples, n_features)
        Input data to embed. Typically a high-dimensional representation of points 
        lying on or near a low-dimensional manifold.

    n_components : int, optional (default=10)
        Number of diffusion components (i.e., eigenvectors) to compute and return. 
        Determines the dimensionality of the resulting embedding.

    Returns
    -------
    dmap : datafold.dynfold.DiffusionMaps
        A fitted DiffusionMaps object from which the embedding can be obtained via 
        `dmap.eigenvectors_`, and diffusion coordinates via `dmap.transform(...)`.
    """
    # use a PCManifold to estimate hyperparameters
    # the attached kernel in PCManifold defaults to a Gaussian kernel
    X_pcm = pfold.PCManifold(vectors)
    X_pcm.optimize_parameters()
    dmap = dfold.DiffusionMaps(
        kernel = pfold.GaussianKernel(
            epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)), 
            n_eigenpairs=n_components, )
    return dmap.fit(X_pcm)



# ========================================
# Supporting Utilities 
# ========================================

def load_word_vectors(bin_path, top_n=10000):
    """
    Load pre-trained Word2Vec vectors from binary .bin file.

    Parameters
    ----------
    bin_path : str
        Path to .bin file.
    top_n : int
        Number of most frequent words to load.

    Returns
    -------
    vectors : np.ndarray
        Word vectors [top_n, dim]
    labels : list[str]
        Word tokens
    """
    model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    vectors = model.vectors[:top_n]
    labels = model.index_to_key[:top_n]
    return vectors, labels

def plot_2d_embedding(X_2d, title="2D Embedding", save_path=None):
    """
    Scatter plot of 2D embedding result.

    Parameters
    ----------
    X_2d : np.ndarray
        2D embedding coordinates.
    labels : list[str]
        Word labels (not shown in plot unless further developed).
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=1, alpha=0.6)
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def dmap_estimate_memory_GB(N=50000, D=100):
    """
    Estimate the total memory usage (in GB) for performing diffusion maps on a dataset.

    This function estimates the memory required to process a dataset with N samples
    and D dimensions using Diffusion Maps. It takes into account both the data matrix 
    (N × D) and the full pairwise distance matrix (N × N), assuming float64 precision 
    (8 bytes per value).

    Parameters
    ----------
    N : int
        Number of data points (samples).

    D : int, optional (default=100)
        Number of features (dimensions) per sample.

    Returns
    -------
    float
        Estimated total memory usage in gigabytes (GB), including:
        - Data matrix: N × D × 8 bytes
        - Pairwise distance matrix: N × N × 8 bytes
    """
    data_mem = N * D * 8
    dist_mem = N * N * 8
    total_mem = data_mem + dist_mem
    return total_mem / (1024**3)

def tsne_estimate_memory_GB(N=50000, D=100, n_neighbors=30, use_float32=True):
    """
    Estimate realistic peak memory usage in GB for running t-SNE on a dataset of size (N, D),
    assuming Barnes-Hut approximation (used by scikit-learn).

    Parameters
    ----------
    N : int
        Number of data points.
    D : int
        Number of features (input dimensions).
    n_neighbors : int
        Approximate number of nearest neighbors per point (used in sparse affinity matrix).
    use_float32 : bool
        Whether data is float32 (4 bytes) or float64 (8 bytes).

    Returns
    -------
    float
        Estimated memory usage in gigabytes (GB).
    """

    bytes_per_float = 4 if use_float32 else 8

    # Input data
    data_mem = N * D * bytes_per_float

    # Sparse affinity matrix: each point has ~n_neighbors entries
    # Note: both P_ij and distances stored (rough approximation)
    sparse_affinity_mem = N * n_neighbors * 2 * bytes_per_float

    # Low-dimensional embedding (usually 2D) and gradient buffer
    emb_mem = N * 2 * bytes_per_float
    grad_mem = N * 2 * bytes_per_float * 2  # momentum, update, etc.

    # Total estimated
    total_bytes = data_mem + sparse_affinity_mem + emb_mem + grad_mem
    return total_bytes / (1024 ** 3)
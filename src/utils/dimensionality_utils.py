from sklearn.manifold import SpectralEmbedding, Isomap, TSNE
#from datafold.dynfold import DiffusionMaps
#from gensim.models import KeyedVectors
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
    t-SNE using scikit-learn (or openTSNE if needed).
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


def run_diffusion_maps(vectors, n_components=2, n_neighbors=10, epsilon='bgh'):
    """
    Diffusion Maps using datafold.
    """
    model = DiffusionMaps(n_eigenpairs=n_components, kernel_scale=epsilon, n_neighbors=n_neighbors)
    return model.fit_transform(vectors)



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

def plot_2d_embedding(X_2d, labels, title="2D Embedding", save_path=None):
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
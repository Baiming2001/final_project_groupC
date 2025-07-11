from sklearn.manifold import SpectralEmbedding, Isomap, TSNE
from datafold.dmap import DiffusionMaps


def run_spectral_embedding(vectors, n_components=2, n_neighbors=20):
    """
    Spectral Embedding using scikit-learn.
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

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances


import numpy as np
from sklearn.neighbors import NearestNeighbors

def continuity(X_high, X_low, n_neighbors=10):
    n = X_high.shape[0]

    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_high)
    high_indices = nn_high.kneighbors(return_distance=False)[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_low)
    low_indices = nn_low.kneighbors(return_distance=False)[:, 1:]

    score = 0
    for i in range(n):
        set_high = set(high_indices[i])
        set_low = set(low_indices[i])
        ui = set_low - set_high
        score += len(ui)
    score = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * score
    return score

def distortion(X_high, X_low):
    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)
    
    # 避免除0
    mask = D_high > 1e-6
    ratio = D_low[mask] / D_high[mask]
    dist = np.abs(ratio - 1).mean()
    return dist


# Step 1: Load vectors from .bin
model = KeyedVectors.load_word2vec_format("src/data/word2vec/text8_vectors.bin", binary=True)
top_n = 1000  # 你可以调整为10000，但越大可视化越慢
words = model.index_to_key[:top_n]
vectors = np.array([model[word] for word in words])

# Step 2: 聚类
kmeans = KMeans(n_clusters=20, random_state=42)
cluster_labels = kmeans.fit_predict(vectors)

# 计算原始高维空间的Silhouette Score
score_highd = silhouette_score(vectors, cluster_labels)
print(f"Silhouette Score (Original High-D): {score_highd:.3f}")
"""
# Step 3: 降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# Step 4: 可视化
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=cluster_labels, cmap='tab10', s=8, alpha=0.7)

# 可选：显示部分单词标签
for i, word in enumerate(words[:300]):  # 只标记前200个防止太挤
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=6, alpha=0.6)

plt.title("t-SNE Visualization of Word2Vec with KMeans Clustering")
plt.axis("off")
plt.tight_layout()
plt.show()
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding

methods = {
    'PCA': PCA(n_components=2),
    't-SNE': TSNE(n_components=2, perplexity=30, random_state=42),
    'Spectral': SpectralEmbedding(n_components=2),
    'Isomap': Isomap(n_components=2),
}

for name, model in methods.items():
    reduced = model.fit_transform(vectors)
    # 降维空间重新聚类
    kmeans_lowd = KMeans(n_clusters=20, random_state=42)
    cluster_labels_lowd = kmeans_lowd.fit_predict(reduced)
    # 计算降维空间的Silhouette Score
    score_lowd = silhouette_score(reduced, cluster_labels_lowd)
    
    print(f"{name}: Silhouette Score (Reduced + Cluster) = {score_lowd:.3f}")
    # Trustworthiness
    tw_score = trustworthiness(vectors, reduced, n_neighbors=10)
    print(f"Trustworthiness: {tw_score:.3f}")
    # Continuity 关注：降维近邻 → 原始空间是否近
    cont_score = continuity(vectors, reduced)
    print(f"Continuity: {cont_score:.3f}")
    #Distortion（扭曲度）判断“距离比例”有没有被压缩或拉伸
    dist_score = distortion(vectors, reduced)
    print(f"Distortion: {dist_score:.3f}")

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels_lowd, cmap='tab10', s=8, alpha=0.7)
    plt.title(f"{name} Visualization + KMeans Clustering")

    # 可选：标注部分单词
    for i, word in enumerate(words[:300]):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=6, alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


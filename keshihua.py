from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load vectors from .bin
model = KeyedVectors.load_word2vec_format("src/data/word2vec/text8_vectors.bin", binary=True)
top_n = 1000  # 你可以调整为10000，但越大可视化越慢
words = model.index_to_key[:top_n]
vectors = np.array([model[word] for word in words])

# Step 2: 聚类
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(vectors)

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

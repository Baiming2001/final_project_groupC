from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# 加载两个 bin 文件
model1 = KeyedVectors.load_word2vec_format("data/word2vec/text8_vectors_train.bin", binary=True)
model2 = KeyedVectors.load_word2vec_format("data/word2vec/text8_vectors_tsne.bin", binary=True)

# 获取词汇表
vocab1 = set(model1.key_to_index.keys())
vocab2 = set(model2.key_to_index.keys())

# 词汇对比
common_words = vocab1 & vocab2
only_in_1 = vocab1 - vocab2
only_in_2 = vocab2 - vocab1

print(f"模型1词数: {len(vocab1)}")
print(f"模型2词数: {len(vocab2)}")
print(f"共同词数: {len(common_words)}")
print(f"只在模型1中的词数: {len(only_in_1)}")
print(f"只在模型2中的词数: {len(only_in_2)}")

# 对比向量差异（用余弦相似度）
sampled_words = list(common_words)[:1000]  # 选前1000个词
similarities = []
for word in sampled_words:
    vec1 = model1[word].reshape(1, -1)
    vec2 = model2[word].reshape(1, -1)
    sim = cosine_similarity(vec1, vec2)[0, 0]
    similarities.append(sim)

avg_sim = np.mean(similarities)
print(f"\n共同词的平均余弦相似度: {avg_sim:.4f}")
print(f"相似度中位数: {np.median(similarities):.4f}")
print(f"最低相似度: {np.min(similarities):.4f}")

# 可视化相似度分布
plt.hist(similarities, bins=30, color='skyblue', edgecolor='black')
plt.title("Cosine Similarity of Shared Words")
plt.xlabel("Similarity")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 可视化具体某些词向量的差异（前5个）
for word in sampled_words[:5]:
    vec1 = model1[word]
    vec2 = model2[word]
    diff = vec2 - vec1
    plt.figure(figsize=(10, 3))
    plt.plot(vec1, label="Model 1")
    plt.plot(vec2, label="Model 2")
    plt.plot(diff, label="Difference", linestyle="--")
    plt.title(f"Vector Comparison for word: '{word}'")
    plt.legend()
    plt.tight_layout()
    plt.show()

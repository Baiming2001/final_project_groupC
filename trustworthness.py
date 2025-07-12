from sklearn.manifold import trustworthiness
from gensim.models import KeyedVectors
import numpy as np

model = KeyedVectors.load_word2vec_format('data/word2vec/text8_vectors_train.bin', binary=True)
words = model.index_to_key[:1000]
high_dim_data = np.array([model[word] for word in words])
trust_tsne = trustworthiness(high_dim_data, tsne_embedding)
trust_spectral = trustworthiness(high_dim_data, spectral_embedding)

print(f"Trustworthiness TSNE: {trust_tsne:.4f}")
print(f"Trustworthiness Spectral Embedding: {trust_spectral:.4f}")


### 再看看
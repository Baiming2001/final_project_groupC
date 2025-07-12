from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("data/word2vec/text8_vectors.bin", binary=True)

# 查看前10个词
for i, word in enumerate(model.index_to_key[:10]):
    print(f"{word}: {model[word][:5]}...")  # 只显示前5维，太长就省略

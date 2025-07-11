import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
text8_path = os.path.join(BASE_DIR, "text8")
save_dir = os.path.join(BASE_DIR, "data", "word2vec")
os.makedirs(save_dir, exist_ok=True)

# Step 1: 加载数据
sentences = LineSentence(text8_path)

# Step 2: 训练 Word2Vec 模型
model = Word2Vec(
    sentences,
    vector_size=300,  # 嵌入维度
    window=5,
    min_count=5,
    sg=1,              # skip-gram (1) 或 CBOW (0)
    workers=4,
    epochs=5
)

# Step 3: 保存模型为 binary 格式（可加载到 gensim 或 numpy）
model.wv.save_word2vec_format("data/word2vec/text8_vectors.bin", binary=True)


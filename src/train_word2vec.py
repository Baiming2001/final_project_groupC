import os
from src.utils.word2vec_utils import *

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text8_path = os.path.join(base_dir, "text8")
    save_path = os.path.join(base_dir, "data", "word2vec", "text8_vectors.bin")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_word2vec(
        text_path=text8_path,
        output_path=save_path
    )
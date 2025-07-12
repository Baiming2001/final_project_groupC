"""
import os
import sys

# 加项目根目录到 sys.path，方便 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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
"""

import os
from src.utils.word2vec_utils import train_word2vec

if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 正确的 text8.txt 路径
    text8_path = os.path.join(project_root, "text8.txt")

    # 保存路径（放到 data/word2vec 下）
    save_path = os.path.join(project_root, "data", "word2vec", "text8_vectors.bin")

    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 检查路径是否存在
    assert os.path.exists(text8_path), f"❌ text8.txt not found at: {text8_path}"

    # 开始训练
    train_word2vec(
        text_path=text8_path,
        output_path=save_path
    )

import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def train_word2vec(
    text_path,
    output_path,
    vector_size=300,
    window=5,
    min_count=5,
    sg=1,
    workers=4,
    epochs=5
):
    """
    Train a Word2Vec model on a text corpus and save the word vectors.

    Parameters
    ----------
    text_path : str
        Path to the input text file.
    output_path : str
        Path to save the binary word vector file.
    vector_size : int
        Dimension of the word embeddings.
    window : int
        Context window size.
    min_count : int
        Minimum word frequency threshold.
    sg : int
        1 = Skip-gram; 0 = CBOW.
    workers : int
        Number of worker threads.
    epochs : int
        Number of training epochs.
    """
    sentences = LineSentence(text_path)
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs
    )
    model.wv.save_word2vec_format(output_path, binary=True)
    print(f"Word2Vec vectors saved to: {output_path}")
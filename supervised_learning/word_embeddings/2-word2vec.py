#!/usr/bin/env python3
"""Creates and trains a Word2Vec model using Gensim."""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build, and train a Gensim Word2Vec model.

    Args:
        sentences: List of tokenized sentences used for training.
        vector_size: Dimensionality of the word vectors.
        min_count: Minimum number of occurrences required for a word.
        window: Maximum distance between a word and its context words.
        negative: Number of negative samples used during training.
        cbow: If True, use CBOW; otherwise, use Skip-gram.
        epochs: Number of training iterations.
        seed: Seed used by the random number generator.
        workers: Number of worker threads used during training.

    Returns:
        The trained Gensim Word2Vec model.
    """
    training_type = 0 if cbow else 1

    model = Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=training_type,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model

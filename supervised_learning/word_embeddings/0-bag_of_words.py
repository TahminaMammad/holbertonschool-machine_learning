#!/usr/bin/env python3
"""Creates a bag-of-words embedding matrix."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Create a bag-of-words embedding matrix.

    Args:
        sentences (list): Sentences to analyze.
        vocab (list, optional): Vocabulary words to use. If None, all
            words found in the sentences are used.

    Returns:
        tuple: A tuple containing:
            embeddings: A NumPy array containing word counts.
            features: A NumPy array containing the vocabulary words.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features

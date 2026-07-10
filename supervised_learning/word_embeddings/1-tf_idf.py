#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix from a list of sentences.

    Args:
        sentences (list): Sentences to analyze.
        vocab (list, optional): Vocabulary words to use. If None, all
            words found in the sentences are used.

    Returns:
        tuple: The TF-IDF embeddings and the features used.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features

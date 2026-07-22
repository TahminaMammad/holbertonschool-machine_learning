#!/usr/bin/env python3
"""Perform semantic search over a corpus of reference documents."""

import os

import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Return the document whose meaning is most similar to sentence."""
    documents = []

    for filename in sorted(os.listdir(corpus_path)):
        file_path = os.path.join(corpus_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, encoding='utf-8') as document:
                documents.append(document.read())

    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
    embeddings = model([sentence] + documents)
    similarities = np.inner(embeddings[0], embeddings[1:])
    best_match = np.argmax(similarities)

    return documents[best_match]

#!/usr/bin/env python3
"""Prepare the TED Portuguese-to-English translation dataset."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load translation data and create Portuguese and English tokenizers."""

    def __init__(self):
        """Load the training and validation splits and train tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create and train Portuguese and English subword tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_corpus = (
            [sentence.numpy().decode('utf-8') for sentence in pt_batch]
            for pt_batch, _ in data.batch(1000)
        )
        en_corpus = (
            [sentence.numpy().decode('utf-8') for sentence in en_batch]
            for _, en_batch in data.batch(1000)
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_corpus, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_corpus, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

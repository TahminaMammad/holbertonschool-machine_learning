#!/usr/bin/env python3
"""Dataset preparation for Portuguese-to-English translation."""

import transformers
from setup :contentReference[oaicite:0]{index=0}Load translation data and create subword tokenizers."""

    def __init__(self):
        """Initialize the datasets and their tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English tokenizers from a dataset."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        portuguese_text = (
            [pt.numpy().decode('utf-8')] for pt, _ in data
        )
        english_text = (
            [en.numpy().decode('utf-8')] for _, en in data
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            portuguese_text,
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            english_text,
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

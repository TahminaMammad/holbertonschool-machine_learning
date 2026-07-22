#!/usr/bin/env python3
"""Load and tokenize the Portuguese-to-English translation dataset."""

import transformers
from setup import load_pt2en


class Dataset:
    """Prepare the TED Portuguese-to-English translation dataset."""

    def __init__(self):
        """Load the training and validation data and create tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers from data."""
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        pt_text = (
            pt.numpy().decode('utf-8') for pt, _ in data
        )
        en_text = (
            en.numpy().decode('utf-8') for _, en in data
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_text, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_text, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

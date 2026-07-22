#!/usr/bin/env python3
"""Prepare a Portuguese-to-English dataset for machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load translation datasets and create subword tokenizers."""

    def __init__(self):
        """Load the datasets and create Portuguese and English tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English tokenizers from the dataset."""
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        pt_corpus = (
            pt.numpy().decode('utf-8') for pt, _ in data
        )
        en_corpus = (
            en.numpy().decode('utf-8') for _, en in data
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_corpus, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_corpus, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode Portuguese and English sentences with boundary tokens."""
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=False
        )
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(
            en_sentence, add_special_tokens=False
        )
        en_tokens.append(self.tokenizer_en.vocab_size + 1)

        return pt_tokens, en_tokens

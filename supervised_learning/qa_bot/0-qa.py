#!/usr/bin/env python3
"""Find answers to questions using a pre-trained BERT model."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Find a text snippet in reference that answers question."""
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load(
        'https://tfhub.dev/see--/bert-uncased-tf2-qa/1'
    )

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]']
    tokens += reference_tokens + ['[SEP]']

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (len(question_tokens) + 2)
    input_type_ids += [1] * (len(reference_tokens) + 1)

    inputs = (input_word_ids, input_mask, input_type_ids)
    input_word_ids, input_mask, input_type_ids = map(
        lambda value: tf.expand_dims(
            tf.convert_to_tensor(value, dtype=tf.int32), 0
        ),
        inputs
    )

    outputs = model([input_word_ids, input_mask, input_type_ids])

    start = tf.argmax(outputs[0][0][1:]) + 1
    end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[start:end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer:
        return None

    return answer

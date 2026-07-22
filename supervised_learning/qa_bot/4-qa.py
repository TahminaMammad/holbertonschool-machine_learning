#!/usr/bin/env python3
"""Answer questions using the most relevant document in a corpus."""

find_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """Answer user questions using documents found in corpus_path."""
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.strip().lower() in exit_words:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)
        answer = find_answer(question, reference)

        if answer is None:
            answer = 'Sorry, I do not understand your question.'

        print('A: {}'.format(answer))

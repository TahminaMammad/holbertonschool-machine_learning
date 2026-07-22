#!/usr/bin/env python3
"""Run an interactive question-answering loop."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer user questions using the supplied reference text."""
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.strip().lower() in exit_words:
            print('A: Goodbye')
            break

        answer = question_answer(question, reference)

        if answer is None:
            answer = 'Sorry, I do not understand your question.'

        print('A: {}'.format(answer))

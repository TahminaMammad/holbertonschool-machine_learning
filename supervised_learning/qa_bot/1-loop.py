#!/usr/bin/env python3
"""Run a simple question-and-answer command-line loop."""

while True:
    question = input("Q: ")

    if question.strip().lower() in {"exit", "quit", "goodbye", "bye"}:
        print("A: Goodbye")
        break

    print("A:")

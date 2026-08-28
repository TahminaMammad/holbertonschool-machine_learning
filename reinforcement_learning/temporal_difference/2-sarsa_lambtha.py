#!/usr/bin/env python3
"""SARSA(lambda) learning with eligibility traces."""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Update a Q table using the SARSA(lambda) algorithm."""
    initial_epsilon = epsilon

    for episode in range(episodes):
        eligibility = np.zeros_like(Q)
        state = env.reset()[0]

        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.random.uniform() < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state])

            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])

            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state, action = next_state, next_action

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q

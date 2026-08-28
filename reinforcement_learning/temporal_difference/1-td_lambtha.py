#!/usr/bin/env python3
"""Temporal-difference learning with eligibility traces."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Estimate state values using the TD(lambda) algorithm."""
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            if terminated or truncated:
                break

    return V

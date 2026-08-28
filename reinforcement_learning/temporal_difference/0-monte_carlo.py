#!/usr/bin/env python3
"""Monte Carlo policy evaluation."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform Monte Carlo policy evaluation."""
    for episode in range(episodes):
        state = env.reset()[0]
        trajectory = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        trajectory = np.array(trajectory, dtype=int)
        returns = 0

        for state, reward in reversed(trajectory):
            returns = reward + gamma * returns

            if state not in trajectory[:episode, 0]:
                V[state] += alpha * (returns - V[state])

    return V

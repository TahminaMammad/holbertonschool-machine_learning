#!/usr/bin/env python3
"""Train an agent using the Monte Carlo policy gradient method."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98,
          show_result=False):
    """Train an agent, optionally render it, and return episode scores."""
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []

        while True:
            if show_result and episode % 1000 == 0:
                env.render()

            action, gradient = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            gradients.append(gradient)
            rewards.append(reward)

            if terminated or truncated:
                break

        score = sum(rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

        discounted_reward = 0
        for index in range(len(rewards) - 1, -1, -1):
            discounted_reward = rewards[index] + gamma * discounted_reward
            weight += alpha * discounted_reward * gradients[index]

    return scores

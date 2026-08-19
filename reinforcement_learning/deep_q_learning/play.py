#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""

import gymnasium as gym
import tensorflow.keras as K

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from train import (
    AtariProcessor,
    KerasRLWrapper,
    WINDOW_LENGTH,
    build_model
)


def play():
    """Load the trained policy and display Breakout games."""
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human"
    )
    env = KerasRLWrapper(env)

    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    processor = AtariProcessor()
    policy = GreedyQPolicy()

    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        processor=processor,
        policy=policy,
        test_policy=policy
    )

    agent.compile(
        K.optimizers.Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    agent.load_weights("policy.h5")

    agent.test(
        env,
        nb_episodes=5,
        visualize=True
    )

    env.close()


if __name__ == "__main__":
    play()

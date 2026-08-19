#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout."""

import gymnasium as gym
import numpy as np
from PIL import Image
import tensorflow.keras as K

from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class KerasRLWrapper(gym.Wrapper):
    """Make a Gymnasium environment compatible with keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Run one step using the old Gym API expected by keras-rl2."""
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render()


class AtariProcessor(Processor):
    """Preprocess Atari observations and rewards."""

    def process_observation(self, observation):
        """Resize an Atari frame and convert it to grayscale."""
        image = Image.fromarray(observation)
        image = image.resize(INPUT_SHAPE).convert("L")
        observation = np.array(image)
        return observation.astype("uint8")

    def process_state_batch(self, batch):
        """Normalize a batch of states."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards to the range [-1, 1]."""
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    """Build the convolutional neural network used by the DQN."""
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = K.models.Sequential()
    model.add(
        K.layers.Permute(
            (2, 3, 1),
            input_shape=input_shape
        )
    )
    model.add(
        K.layers.Conv2D(
            32,
            (8, 8),
            strides=(4, 4),
            activation="relu"
        )
    )
    model.add(
        K.layers.Conv2D(
            64,
            (4, 4),
            strides=(2, 2),
            activation="relu"
        )
    )
    model.add(
        K.layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation="relu"
        )
    )
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation="relu"))
    model.add(K.layers.Dense(nb_actions, activation="linear"))

    return model


def train():
    """Train the DQN agent and save its policy network."""
    env = gym.make("ALE/Breakout-v5")
    env = KerasRLWrapper(env)

    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )

    processor = AtariProcessor()

    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    agent.compile(
        K.optimizers.Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    agent.fit(
        env,
        nb_steps=1000000,
        visualize=False,
        verbose=2,
        log_interval=10000
    )

    agent.save_weights("policy.h5", overwrite=True)
    env.close()


if __name__ == "__main__":
    train()

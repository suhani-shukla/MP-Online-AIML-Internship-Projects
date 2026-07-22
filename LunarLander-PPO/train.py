"""
train.py
--------
Trains a PPO agent on the Gymnasium LunarLander environment
using Stable-Baselines3.

Supports RESUMABLE training: if a saved model already exists,
it loads it and continues training instead of starting fresh.
This lets you train in smaller chunks across multiple sessions
instead of running all TOTAL_TIMESTEPS in one sitting.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config


def build_env():
    """
    Creates and wraps the LunarLander environment.
    render_mode=None during training for speed (no rendering overhead).
    Monitor logs per-episode reward/length to LOGS_DIR for later plotting.
    """
    env = gym.make(config.ENV_ID, render_mode=None)
    env = Monitor(env, filename=config.LOGS_DIR)
    return env


def build_model(env):
    """Creates a fresh PPO model using hyperparameters from config.py."""
    model = PPO(
        policy=config.POLICY,
        env=env,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        batch_size=config.BATCH_SIZE,
        n_steps=config.N_STEPS,
        n_epochs=config.N_EPOCHS,
        clip_range=config.CLIP_RANGE,
        verbose=config.VERBOSE,
        tensorboard_log=config.LOGS_DIR,
    )
    return model


def train(chunk_timesteps=None):
    """
    Trains the PPO agent in one chunk of timesteps.

    If a saved model already exists at config.MODEL_PATH, it is loaded
    and training continues from there (resume). Otherwise, a fresh
    model is created.

    Args:
        chunk_timesteps (int, optional): Number of timesteps to train
            in THIS run. Defaults to config.CHUNK_TIMESTEPS if not given.
    """
    if chunk_timesteps is None:
        chunk_timesteps = config.CHUNK_TIMESTEPS

    env = build_env()
    model_file = config.MODEL_PATH + ".zip"

    if os.path.exists(model_file):
        print(f"Existing model found at {model_file}. Resuming training...")
        model = PPO.load(config.MODEL_PATH, env=env)
    else:
        print("No existing model found. Creating a new PPO model...")
        model = build_model(env)

    print(f"Training for {chunk_timesteps} more timesteps...")
    # reset_num_timesteps=False keeps the internal timestep counter
    # continuous across chunks (important for correct logging/scheduling)
    model.learn(
        total_timesteps=chunk_timesteps,
        reset_num_timesteps=False,
    )

    print(f"Saving model to {model_file}")
    model.save(config.MODEL_PATH)

    env.close()
    print("Chunk complete. Environment closed.")


if __name__ == "__main__":
    train()
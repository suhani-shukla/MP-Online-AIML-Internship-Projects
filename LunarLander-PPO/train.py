"""
train.py
--------
Trains a PPO agent on the Gymnasium LunarLander environment
using Stable-Baselines3.

Supports RESUMABLE training: if a saved model already exists,
it loads it and continues training instead of starting fresh.

Uses two callbacks during training:
    - CheckpointCallback: saves the model periodically, protecting
      against lost progress if a training run is interrupted.
    - EvalCallback: periodically evaluates the current policy on a
      separate environment and saves the best-performing version
      separately from the "latest" model.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

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


def build_eval_env():
    """
    Separate environment used only by EvalCallback for periodic
    evaluation during training. Kept independent from the training
    env so evaluation episodes don't interfere with rollout collection.
    """
    env = gym.make(config.ENV_ID, render_mode=None)
    env = Monitor(env)
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


def build_callbacks():
    """
    Builds and returns the list of callbacks used during training:

    1. CheckpointCallback: saves a snapshot of the model every
       CHECKPOINT_FREQ timesteps into models/checkpoints/, named
       with the timestep count -- so you can roll back to any point.

    2. EvalCallback: every EVAL_FREQ timesteps, runs a handful of
       evaluation episodes on a separate env and saves the model
       to models/best_model/ ONLY if it beats the previous best
       mean reward. This protects your best policy even if later
       training makes things temporarily worse.
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.CHECKPOINT_DIR,
        name_prefix="ppo_lunarlander",
    )

    eval_env = build_eval_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.BEST_MODEL_DIR,
        log_path=config.LOGS_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES_CALLBACK,
        deterministic=True,
        render=False,
    )

    return [checkpoint_callback, eval_callback]


def train(chunk_timesteps=None):
    """
    Trains the PPO agent in one chunk of timesteps, with checkpointing
    and periodic evaluation active throughout.

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

    callbacks = build_callbacks()

    print(f"Training for {chunk_timesteps} more timesteps...")
    model.learn(
        total_timesteps=chunk_timesteps,
        reset_num_timesteps=False,
        callback=callbacks,
    )

    print(f"Saving model to {model_file}")
    model.save(config.MODEL_PATH)

    env.close()
    print("Chunk complete. Environment closed.")


if __name__ == "__main__":
    train()
"""
visualize_agent.py
-------------------
Loads a trained PPO model and renders it live so you can visually
inspect the agent's landing behavior.

This is for VISUAL INSPECTION only -- no reward statistics are
computed here (that's evaluate.py's job). Use this after training
to actually watch the rocket:
    - maintain balance
    - reduce descent speed
    - align with the landing pad
    - touch down gently
    - remain upright

IMPORTANT: render_mode="human" is only used here, never during
training, since rendering every frame would drastically slow down
the training loop.
"""

import os
import time
import gymnasium as gym
from stable_baselines3 import PPO

import config


def build_render_env():
    """
    Creates the LunarLander environment with a live rendering window.
    """
    env = gym.make(config.ENV_ID, render_mode="human")
    return env


def visualize(n_episodes=None):
    """
    Loads the saved model and runs it for n_episodes, rendering each
    step live so the agent's behavior can be observed on screen.

    Args:
        n_episodes (int, optional): Number of episodes to visualize.
            Defaults to config.RENDER_EPISODES if not provided.
    """
    if n_episodes is None:
        n_episodes = config.RENDER_EPISODES

    model_file = config.MODEL_PATH + ".zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"No trained model found at {model_file}. "
            f"Run train.py first to produce a saved model."
        )

    print(f"Loading model from {model_file}")
    model = PPO.load(config.MODEL_PATH)

    env = build_render_env()

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            # deterministic=True: always pick the policy's best action,
            # rather than sampling -- gives consistent, repeatable behavior
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Slow down playback slightly so the animation is watchable
            time.sleep(config.RENDER_SLEEP)

        print(f"Episode {episode}/{n_episodes} -- Reward = {total_reward:.2f}")

    env.close()
    print("Visualization complete. Environment closed.")


if __name__ == "__main__":
    visualize()
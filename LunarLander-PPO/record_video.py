"""
record_video.py
----------------
Loads a trained PPO model and records its behavior as MP4 videos
using Gymnasium's RecordVideo wrapper.

Unlike visualize_agent.py (which opens a live window), this uses
render_mode="rgb_array" -- frames are rendered to arrays and encoded
into video files instead of displayed on screen. This is useful for:
    - sharing the agent's behavior without needing a live session
    - reviewing landings later
    - including demo clips in a report or README

Videos are saved to config.VIDEOS_DIR.
"""

import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

import config


def build_video_env(n_episodes):
    """
    Creates the LunarLander environment wrapped with RecordVideo.

    render_mode="rgb_array": frames are captured as arrays (not shown
    live) so they can be encoded into an MP4 file.

    episode_trigger=lambda ep: True records every episode; you could
    change this to record only every Nth episode if you run many.
    """
    env = gym.make(config.ENV_ID, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=config.VIDEOS_DIR,
        episode_trigger=lambda episode: True,
        name_prefix="ppo_lunarlander",
    )
    return env


def record(n_episodes=None):
    """
    Loads the saved model and runs it for n_episodes, recording each
    episode as a separate MP4 file in config.VIDEOS_DIR.

    Args:
        n_episodes (int, optional): Number of episodes to record.
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

    env = build_video_env(n_episodes)

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode}/{n_episodes} recorded -- Reward = {total_reward:.2f}")

    env.close()
    print(f"Videos saved in: {config.VIDEOS_DIR}")


if __name__ == "__main__":
    record()
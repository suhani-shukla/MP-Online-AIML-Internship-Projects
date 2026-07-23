"""Record video of a trained PPO agent playing CartPole-v1."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
VIDEOS_DIR = ROOT_DIR / "videos"


def record(model_path: Path, n_episodes: int = 5, name_prefix: str = "ppo_cartpole"):
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(VIDEOS_DIR),
        name_prefix=name_prefix,
        episode_trigger=lambda ep: True,
    )
    model = PPO.load(str(model_path))

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    print(f"Saved {n_episodes} episode videos to: {VIDEOS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record video of trained agent")
    parser.add_argument(
        "--model-path", type=str, default=str(MODELS_DIR / "best_model" / "best_model.zip")
    )
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record(Path(args.model_path), n_episodes=args.episodes)

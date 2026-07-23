"""Watch a trained PPO agent play CartPole-v1 live."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"


def visualize(model_path: Path, n_episodes: int = 5):
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load(str(model_path))

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode}: reward = {total_reward}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained agent")
    parser.add_argument(
        "--model-path", type=str, default=str(MODELS_DIR / "best_model" / "best_model.zip")
    )
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(Path(args.model_path), n_episodes=args.episodes)

"""Evaluate a trained PPO model on CartPole-v1."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"


def evaluate(model_path: Path, n_episodes: int = 20, env_id: str = "CartPole-v1"):
    env = gym.make(env_id)
    model = PPO.load(str(model_path), env=env)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )
    env.close()
    return mean_reward, std_reward


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument(
        "--model-path", type=str, default=str(MODELS_DIR / "best_model" / "best_model.zip")
    )
    parser.add_argument("--episodes", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mean_reward, std_reward = evaluate(Path(args.model_path), n_episodes=args.episodes)
    print(f"Mean reward over {args.episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Solved (>=475 avg reward): {mean_reward >= 475}")

"""
Stage 2: Basic PPO training on CartPole-v1.

Trains for a fixed number of timesteps and saves the resulting model.
No checkpointing / early stopping yet — that's added in Stage 3.
"""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from cartpole_rl.config import PPOConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"


def train(config: PPOConfig) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make(config.env_id))
    model = PPO("MlpPolicy", env, seed=config.seed, verbose=1)
    model.learn(total_timesteps=config.total_timesteps)

    model_path = MODELS_DIR / f"{config.model_name}.zip"
    model.save(str(model_path))

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Post-training eval: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    return model_path


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="ppo_cartpole")
    args = parser.parse_args()
    return PPOConfig(total_timesteps=args.timesteps, seed=args.seed, model_name=args.model_name)


if __name__ == "__main__":
    cfg = parse_args()
    path = train(cfg)
    print(f"Model saved to: {path}")

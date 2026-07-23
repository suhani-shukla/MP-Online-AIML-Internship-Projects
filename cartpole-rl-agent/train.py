"""Train a PPO agent to solve CartPole-v1."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor

from config import PPOConfig

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"


def train(config: PPOConfig) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make(config.env_id), filename=str(LOGS_DIR / "monitor.csv"))
    eval_env = Monitor(gym.make(config.env_id))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        seed=config.seed,
        verbose=1,
        tensorboard_log=str(LOGS_DIR),
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=config.reward_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=config.eval_freq,
        best_model_save_path=str(MODELS_DIR / "best_model"),
        log_path=str(LOGS_DIR),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=config.model_name,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    final_path = MODELS_DIR / f"{config.model_name}.zip"
    model.save(str(final_path))

    env.close()
    eval_env.close()
    return final_path


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PPOConfig(total_timesteps=args.timesteps, seed=args.seed)


if __name__ == "__main__":
    cfg = parse_args()
    path = train(cfg)
    print(f"Training complete. Model saved to: {path}")

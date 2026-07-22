"""
Stage 3: Full PPO training with checkpointing, periodic evaluation,
and early stopping once the environment is considered solved.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env

from cartpole_rl.config import PPOConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"


def make_env(config: PPOConfig, log_subdir: str):
    monitor_dir = LOGS_DIR / log_subdir
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return make_vec_env(
        config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        monitor_dir=str(monitor_dir),
    )


def train(config: PPOConfig) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_env(config, log_subdir="train_monitor")
    eval_env = make_env(config, log_subdir="eval_monitor")

    model = PPO(
        "MlpPolicy",
        train_env,
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
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=config.reward_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=max(config.eval_freq // config.n_envs, 1),
        best_model_save_path=str(MODELS_DIR / "best_model"),
        log_path=str(LOGS_DIR / "eval"),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.checkpoint_freq // config.n_envs, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix=config.model_name,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    final_path = MODELS_DIR / f"{config.model_name}_final.zip"
    model.save(str(final_path))

    train_env.close()
    eval_env.close()
    return final_path


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="ppo_cartpole")
    args = parser.parse_args()
    return PPOConfig(
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    cfg = parse_args()
    saved_path = train(cfg)
    print(f"Training complete. Final model saved to: {saved_path}")

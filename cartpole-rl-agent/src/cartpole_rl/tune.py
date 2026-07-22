"""
Stage 5: Lightweight hyperparameter sweep for PPO on CartPole-v1.

Trains several short runs with different hyperparameter combinations
and reports which performed best, to inform the final training config.
"""

import csv
import itertools
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from cartpole_rl.config import PPOConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "logs"

LEARNING_RATES = [1e-3, 3e-4, 1e-4]
N_STEPS_OPTIONS = [64, 128, 256]
TUNE_TIMESTEPS = 20_000


def run_single_trial(learning_rate: float, n_steps: int, seed: int = 0) -> float:
    config = PPOConfig(
        total_timesteps=TUNE_TIMESTEPS,
        learning_rate=learning_rate,
        n_steps=n_steps,
        seed=seed,
        n_envs=2,
    )
    env = make_vec_env(config.env_id, n_envs=config.n_envs, seed=config.seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    env.close()
    return mean_reward


def run_sweep():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = LOGS_DIR / "tuning_results.csv"
    results = []

    for lr, n_steps in itertools.product(LEARNING_RATES, N_STEPS_OPTIONS):
        print(f"Trial: learning_rate={lr}, n_steps={n_steps}")
        mean_reward = run_single_trial(lr, n_steps)
        results.append({"learning_rate": lr, "n_steps": n_steps, "mean_reward": mean_reward})
        print(f"  -> mean_reward={mean_reward:.2f}")

    results.sort(key=lambda r: r["mean_reward"], reverse=True)

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["learning_rate", "n_steps", "mean_reward"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBest config: {results[0]}")
    print(f"Full results saved to: {results_path}")
    return results


if __name__ == "__main__":
    run_sweep()

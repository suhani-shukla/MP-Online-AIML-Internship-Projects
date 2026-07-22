"""
evaluate.py
-----------
Loads a trained PPO model and evaluates its performance on the
LunarLander environment using Stable-Baselines3's evaluate_policy.

This gives a quantitative measure (mean reward, std reward) of how
well the agent has learned, without any rendering — fast and
suitable for repeated checks after every training chunk.

Interpretation guide (from project reference):
    Mean Reward     Performance
    Below 0         Poor
    0 - 100         Moderate
    100 - 200       Good
    Above 200       Excellent
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import config


def build_eval_env():
    """
    Creates the evaluation environment.

    render_mode=None here too -- evaluation is about measuring
    reward statistics quickly, not watching the agent. Visual
    inspection is handled separately by visualize_agent.py (Stage 5).
    """
    env = gym.make(config.ENV_ID, render_mode=None)
    env = Monitor(env)
    return env


def evaluate():
    """
    Loads the saved model and runs evaluate_policy over
    config.N_EVAL_EPISODES episodes, printing mean and std reward.
    """
    model_file = config.MODEL_PATH + ".zip"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"No trained model found at {model_file}. "
            f"Run train.py first to produce a saved model."
        )

    print(f"Loading model from {model_file}")
    model = PPO.load(config.MODEL_PATH)

    env = build_eval_env()

    print(f"Evaluating over {config.N_EVAL_EPISODES} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,  # use the learned policy's best action, no random sampling
    )

    env.close()

    print("\n--- Evaluation Results ---")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward:  {std_reward:.2f}")
    print(interpret_reward(mean_reward))

    return mean_reward, std_reward


def interpret_reward(mean_reward):
    """Returns a human-readable performance label for a given mean reward."""
    if mean_reward < 0:
        return "Performance: Poor"
    elif mean_reward < 100:
        return "Performance: Moderate"
    elif mean_reward < 200:
        return "Performance: Good"
    else:
        return "Performance: Excellent"


if __name__ == "__main__":
    evaluate()
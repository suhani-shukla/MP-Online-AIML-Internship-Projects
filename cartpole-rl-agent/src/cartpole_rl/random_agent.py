"""
Stage 1: Random-action baseline agent for CartPole-v1.

Runs the environment with randomly sampled actions to establish
a baseline performance before training an actual RL agent (PPO).
"""

import csv
from pathlib import Path

import gymnasium as gym

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def run_random_episodes(num_episodes: int = 10, seed: int = 42, render: bool = False) -> list[float]:
    """Run CartPole-v1 for num_episodes using random actions.

    Returns a list of total reward per episode.
    """
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    episode_rewards: list[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards


def save_rewards_to_csv(rewards: list[float], filename: str = "random_agent_baseline.csv") -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filepath = LOG_DIR / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i, reward])

    return filepath


if __name__ == "__main__":
    rewards = run_random_episodes(num_episodes=20)
    avg_reward = sum(rewards) / len(rewards)

    print(f"Random agent baseline over {len(rewards)} episodes:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Min reward: {min(rewards):.2f}")
    print(f"  Max reward: {max(rewards):.2f}")

    log_path = save_rewards_to_csv(rewards)
    print(f"Saved episode rewards to: {log_path}")

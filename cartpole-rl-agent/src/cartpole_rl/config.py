"""Training configuration for the CartPole PPO agent."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 50_000
    seed: int = 42
    model_name: str = "ppo_cartpole"

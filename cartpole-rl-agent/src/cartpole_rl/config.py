"""Training configuration for the CartPole PPO agent."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    seed: int = 42
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    reward_threshold: float = 475.0
    eval_freq: int = 2000
    checkpoint_freq: int = 5000
    model_name: str = "ppo_cartpole"

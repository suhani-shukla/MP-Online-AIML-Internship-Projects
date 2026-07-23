"""Plot the training learning curve from the Monitor CSV log."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs"
GRAPHS_DIR = ROOT_DIR / "graphs"


def plot_learning_curve(output_name: str = "training_rewards.png"):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOGS_DIR / "monitor.csv", skiprows=1)
    df["cumulative_timesteps"] = df["l"].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(df["cumulative_timesteps"], df["r"], alpha=0.3, label="Episode reward")
    plt.plot(
        df["cumulative_timesteps"],
        df["r"].rolling(window=20, min_periods=1).mean(),
        label="20-episode rolling mean",
        linewidth=2,
    )
    plt.axhline(y=475, color="green", linestyle="--", label="Solved threshold (475)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training on CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = GRAPHS_DIR / output_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved learning curve to: {output_path}")


if __name__ == "__main__":
    plot_learning_curve()

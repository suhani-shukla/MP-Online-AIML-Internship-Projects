"""
Stage 4: Plot the training learning curve from SB3 Monitor CSV logs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "logs"


def load_monitor_logs(monitor_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_file in sorted(monitor_dir.glob("*.monitor.csv")):
        df = pd.read_csv(csv_file, skiprows=1)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No monitor CSV files found in {monitor_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined["cumulative_timesteps"] = combined["l"].cumsum()
    return combined


def plot_learning_curve(monitor_subdir: str = "train_monitor", output_name: str = "learning_curve.png"):
    df = load_monitor_logs(LOGS_DIR / monitor_subdir)

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

    output_path = LOGS_DIR / output_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved learning curve to: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_learning_curve()

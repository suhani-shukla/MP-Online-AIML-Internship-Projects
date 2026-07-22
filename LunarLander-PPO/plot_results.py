"""
plot_results.py
----------------
Parses the Monitor training logs (CSV) produced during train.py runs
and plots the agent's reward progress over episodes using Matplotlib.

Stable-Baselines3's Monitor wrapper writes a CSV file with per-episode
reward (`r`) and length (`l`). This script loads that file and plots:
    1. Raw episode rewards over time
    2. A rolling average (smoothed curve) to reveal the learning trend

A well-trained PPO agent typically shows increasing average reward
and decreasing variance over time -- this plot lets you visually
confirm that.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

import config


def load_monitor_data():
    """
    Loads and concatenates all Monitor CSV log files found in LOGS_DIR.

    Monitor files are named like 'monitor.csv' (or with a timestamp
    prefix depending on SB3 version) and have a special header row
    starting with '#' that must be skipped.
    """
    csv_files = glob.glob(os.path.join(config.LOGS_DIR, "*.monitor.csv"))
    if not csv_files:
        # Fallback: some SB3 versions name it just "monitor.csv"
        csv_files = glob.glob(os.path.join(config.LOGS_DIR, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No monitor log files found in {config.LOGS_DIR}. "
            f"Make sure train.py has been run at least once."
        )

    print(f"Found {len(csv_files)} log file(s): {csv_files}")

    dataframes = []
    for file in csv_files:
        # skiprows=1 skips the '#' comment header line Monitor writes
        df = pd.read_csv(file, skiprows=1)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)
    return data


def plot_rewards(data, window=20):
    """
    Plots raw episode rewards and a rolling average over them.

    Args:
        data (pd.DataFrame): Must contain a column 'r' (episode reward).
        window (int): Number of episodes to average over for smoothing.
    """
    rewards = data["r"]
    episodes = range(1, len(rewards) + 1)
    rolling_avg = rewards.rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward", color="steelblue")
    plt.plot(episodes, rolling_avg, label=f"Rolling Average (window={window})", color="darkorange", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Progress on LunarLander")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = os.path.join(config.GRAPHS_DIR, "training_rewards.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    plt.show()


def main():
    data = load_monitor_data()
    plot_rewards(data)


if __name__ == "__main__":
    main()
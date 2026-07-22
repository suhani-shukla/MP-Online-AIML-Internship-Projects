# LunarLander-PPO

An autonomous agent that learns to safely land a spacecraft on a
landing pad using **Proximal Policy Optimization (PPO)**, trained
via reinforcement learning rather than hand-coded rules.

Part of a multi-project RL monorepo. This folder is self-contained
except for the shared Python virtual environment at the repo root.

---

## Objective

Train an agent to control a lander's engines (main, left, right, or
none) purely from reward feedback, until it consistently achieves
safe, stable, fuel-efficient landings on the Gymnasium `LunarLander`
environment.

---

## Tech Stack

- Python 3.x
- [Gymnasium](https://gymnasium.farama.org/) (`LunarLander-v3`, Box2D physics)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO implementation)
- PyTorch (neural network backend)
- NumPy
- Matplotlib / Pandas (plotting training curves)
- MoviePy (video recording)

---

## Project Structure


```
LunarLander-PPO/
├── config.py                  # Centralized hyperparameters and project paths
├── train.py                   # Resumable, chunked PPO training with callbacks
├── evaluate.py                # Numeric performance evaluation (mean/std reward)
├── visualize_agent.py         # Live rendering of the trained agent
├── record_video.py            # Records agent behavior as MP4
├── plot_results.py            # Generates training reward plots
├── requirements.txt           # Project-specific dependencies
├── models/
│   ├── ppo_lunarlander.zip    # Latest trained model (updated after each training run)
│   ├── checkpoints/           # Periodic training checkpoints
│   └── best_model/            # Best-performing model during training
├── logs/                      # Monitor CSVs and evaluation logs (ignored by Git)
├── videos/                    # Recorded episode videos (ignored by Git)
└── graphs/                    # Generated plots (e.g., training_rewards.png)
```

## Directory Overview

| File / Folder | Purpose |
|---------------|---------|
| `config.py` | Stores all configurable hyperparameters and file paths in one place. |
| `train.py` | Trains the PPO agent, supports checkpointing and resuming training. |
| `evaluate.py` | Evaluates the trained agent over multiple episodes and reports performance metrics. |
| `visualize_agent.py` | Runs the trained agent with live rendering for visual inspection. |
| `record_video.py` | Records gameplay videos of the trained agent for demonstrations. |
| `plot_results.py` | Reads training logs and generates reward/performance graphs. |
| `requirements.txt` | Lists Python dependencies required for this project. |
| `models/` | Stores trained models, checkpoints, and the best-performing model. |
| `logs/` | Contains Monitor logs and evaluation outputs (not committed to Git). |
| `videos/` | Stores generated MP4 recordings (not committed to Git). |
| `graphs/` | Stores generated plots and visualizations (not committed to Git). |

---

## Setup

From the **repo root** (shared virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r LunarLander-PPO/requirements.txt
```

---

## Usage

All commands below are run from inside `LunarLander-PPO/`.

### 1. Train the agent (resumable, chunked)
```bash
python train.py
```
- Trains `config.CHUNK_TIMESTEPS` timesteps per run (default: 50,000).
- Automatically resumes from the last saved model if one exists.
- Run this multiple times to reach the full `config.TOTAL_TIMESTEPS`.
- Saves periodic checkpoints (`models/checkpoints/`) and the best
  model seen so far (`models/best_model/`) via callbacks.

### 2. Evaluate performance
```bash
python evaluate.py
```
Runs `config.N_EVAL_EPISODES` deterministic episodes and reports:

| Mean Reward | Performance |
|---|---|
| Below 0 | Poor |
| 0–100 | Moderate |
| 100–200 | Good |
| Above 200 | Excellent |

### 3. Watch the agent live
```bash
python visualize_agent.py
```
Opens a live rendering window and runs `config.RENDER_EPISODES` episodes.

### 4. Record episodes as video
```bash
python record_video.py
```
Saves MP4s to `videos/` — one per episode.

### 5. Plot training progress
```bash
python plot_results.py
```
Parses Monitor logs and saves a reward-curve plot to `graphs/training_rewards.png`.

---

## Environment Details

- **Observation space:** `Box(-inf, inf, (8,), float32)` — position,
  velocity, angle, angular velocity, and leg-contact flags.
- **Action space:** `Discrete(4)` — do nothing, fire left engine,
  fire main engine, fire right engine.
- **Reward shaping:** rewards proximity to the pad, upright
  orientation, gentle descent, and leg contact; penalizes crashes,
  drifting, excessive fuel use, and instability.

---

## Notes on Results

Training shows the typical PPO learning signature for LunarLander:
early episodes are dominated by crashes, and as training progresses,
successful landings (200+ reward) become more frequent. Reward
variance can stay high mid-training since the policy may perform
excellently on some initial conditions and poorly on others before
fully converging — the rolling average in `plot_results.py` is the
best signal of real progress, more so than any single episode.

---

## Future Improvements

- Hyperparameter tuning (entropy coefficient, `n_steps`, learning rate)
  to reduce reward variance further.
- Automated hyperparameter search (e.g. Optuna).
- Continuous action space variant (`LunarLanderContinuous-v3`).
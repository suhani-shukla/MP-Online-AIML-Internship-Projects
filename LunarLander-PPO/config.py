"""
config.py
----------
Centralized configuration for the LunarLander PPO project.
All paths, hyperparameters, and environment settings live here
so every script (train.py, evaluate.py, visualize_agent.py, etc.)
stays consistent and easy to tune from one place.
"""

import os

# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------
# Base directory = this project's own folder (not repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")

MODEL_NAME = "ppo_lunarlander"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)  # SB3 appends .zip automatically

# Ensure required directories exist even if .gitkeep files are removed
for directory in [MODELS_DIR, LOGS_DIR, VIDEOS_DIR, GRAPHS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------
ENV_ID = "LunarLander-v3"   # fallback to "LunarLander-v2" if v3 unavailable in your gymnasium version

# ---------------------------------------------------------
# PPO Hyperparameters
# ---------------------------------------------------------
POLICY = "MlpPolicy"
LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 64
N_STEPS = 2048
N_EPOCHS = 10
CLIP_RANGE = 0.2
VERBOSE = 1

# ---------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------
TOTAL_TIMESTEPS = 500_000

# ---------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------
N_EVAL_EPISODES = 20

# ---------------------------------------------------------
# Visualization Configuration
# ---------------------------------------------------------
RENDER_EPISODES = 5
RENDER_SLEEP = 0.02  # seconds between frames, for smoother visual playback

# ---------------------------------------------------------
# Chunked Training Configuration
# ---------------------------------------------------------
CHUNK_TIMESTEPS = 50_000   # train this many timesteps per run
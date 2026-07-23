# CartPole RL Agent

Training a PPO agent (Stable-Baselines3) to solve the classic
CartPole-v1 control task from Gymnasium.

## Setup

Uses the shared venv at the repo root.

    source ../.venv/Scripts/activate
    pip install -r requirements.txt

## Usage

Train (stops early once solved, avg reward >= 475):

    python train.py --timesteps 200000

Evaluate the best saved model:

    python evaluate.py --model-path models/best_model/best_model.zip

Plot the learning curve:

    python plot_results.py

Watch the agent play live:

    python visualize_agent.py

Record videos of the agent:

    python record_video.py

## Results

_Fill in after training:_
- PPO average reward: TBD
- Timesteps to solve: TBD

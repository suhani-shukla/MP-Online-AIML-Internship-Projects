# CartPole RL Agent

Training a PPO agent (Stable-Baselines3) to solve the classic
CartPole-v1 control task from Gymnasium.

## Setup

This project uses the shared venv at the repo root.

    source ../.venv/Scripts/activate
    pip install -r requirements.txt

## Project stages

0. Project scaffolding
1. Random-action baseline agent
2. Basic PPO training script
3. Checkpointing, eval callback, early stopping
4. Evaluation and learning-curve visualization
5. Hyperparameter tuning sweep
6. Final polish (this stage)

## Usage

Train (stops early once solved, i.e. avg reward >= 475):

    python src/cartpole_rl/train.py --timesteps 200000

Evaluate the best saved model:

    python src/cartpole_rl/evaluate.py --model-path models/best_model/best_model.zip

Plot the learning curve:

    python src/cartpole_rl/plot_results.py

Run the hyperparameter sweep:

    python src/cartpole_rl/tune.py

Run all tests:

    pytest tests/ -v

## Results

_Fill in after your first full training run:_
- Baseline (random agent) average reward: ~20
- PPO average reward after training: TBD
- Timesteps to solve: TBD

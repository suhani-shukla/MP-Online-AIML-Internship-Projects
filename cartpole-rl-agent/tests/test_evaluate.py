from pathlib import Path

from stable_baselines3 import PPO

from cartpole_rl.evaluate import evaluate


def test_evaluate_runs_and_returns_stats(tmp_path):
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, seed=0)
    model.learn(total_timesteps=256)

    model_path = tmp_path / "test_model.zip"
    model.save(str(model_path))
    env.close()

    mean_reward, std_reward = evaluate(model_path, n_episodes=3)

    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)

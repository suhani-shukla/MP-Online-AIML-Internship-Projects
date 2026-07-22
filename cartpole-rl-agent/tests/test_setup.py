import gymnasium as gym
import stable_baselines3


def test_gymnasium_import():
    assert gym.__version__


def test_stable_baselines3_import():
    assert stable_baselines3.__version__


def test_cartpole_env_creation():
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    assert obs.shape == (4,)
    assert env.action_space.n == 2
    env.close()

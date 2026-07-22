from cartpole_rl.random_agent import run_random_episodes


def test_run_random_episodes_returns_correct_length():
    rewards = run_random_episodes(num_episodes=3, seed=0)
    assert len(rewards) == 3


def test_rewards_are_positive_floats():
    rewards = run_random_episodes(num_episodes=2, seed=0)
    for r in rewards:
        assert isinstance(r, float)
        assert r > 0

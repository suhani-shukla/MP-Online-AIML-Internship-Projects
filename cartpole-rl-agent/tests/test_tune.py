from cartpole_rl.tune import run_single_trial


def test_run_single_trial_returns_float():
    reward = run_single_trial(learning_rate=3e-4, n_steps=64, seed=0)
    assert isinstance(reward, float)

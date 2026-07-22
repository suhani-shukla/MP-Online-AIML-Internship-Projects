from cartpole_rl.config import PPOConfig
from cartpole_rl.train import train


def test_train_smoke(tmp_path, monkeypatch):
    import cartpole_rl.train as train_module
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "LOGS_DIR", tmp_path / "logs")

    config = PPOConfig(
        total_timesteps=512,
        n_envs=2,
        eval_freq=256,
        checkpoint_freq=256,
        model_name="smoke_test_model",
    )
    model_path = train(config)

    assert model_path.exists()
    assert (tmp_path / "models" / "checkpoints").exists()

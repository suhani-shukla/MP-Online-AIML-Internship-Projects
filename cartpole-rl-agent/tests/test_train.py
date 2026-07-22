from cartpole_rl.config import PPOConfig
from cartpole_rl.train import train


def test_train_smoke(tmp_path, monkeypatch):
    import cartpole_rl.train as train_module
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    config = PPOConfig(total_timesteps=256, seed=0, model_name="smoke_test_model")
    model_path = train(config)

    assert model_path.exists()
    assert model_path.name == "smoke_test_model.zip"

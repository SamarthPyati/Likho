from likho.config import Settings


def test_mock_models_reads_env(monkeypatch):
    monkeypatch.setenv("MOCK_MODELS", "1")
    assert Settings().mock_models is True
    monkeypatch.setenv("MOCK_MODELS", "0")
    assert Settings().mock_models is False


def test_hf_token_is_secret(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    settings = Settings()
    assert settings.hf_token is not None
    assert "hf_abc" not in repr(settings)
    assert settings.hf_token.get_secret_value() == "hf_abc"

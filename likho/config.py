"""Process-wide settings, read from the environment and `.env`."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mock_models: bool = False  # MOCK_MODELS=1 → deterministic stubs, no GPU, no downloads
    hf_token: SecretStr | None = None  # HF_TOKEN; IndicTrans2 is gated
    log_level: str = "INFO"
    llm_tiebreak: bool = True  # LLM_TIEBREAK=0 disables the tagger's Param-1 call (ablation 3)
    llm_model_id: str = "bharatgenai/Param-1-2.9B-Instruct"
    llm_seed: int = 3235  # sampling seed, reset per call (reproducible eval)
    language: str = "hi"  # Hindi only ships (PRD §6); kept as config, not a constant


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

"""Application configuration via environment variables."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment and .env file."""

    GROQ_API_KEY: str = ""
    API_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    USE_CACHE: bool = False
    USE_LOCAL_LLM: bool = False
    LOCAL_LLM_URL: str = "http://localhost:11434"
    MAX_FILE_SIZE_MB: int = 50
    REQUEST_TIMEOUT_SECONDS: int = 300
    CORS_ORIGINS: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("GROQ_API_KEY", "API_KEY", mode="after")
    @classmethod
    def validate_required_keys(cls, v: str, info) -> str:
        """Ensure critical API keys are provided."""
        if not v or not v.strip():
            field_name = info.field_name
            raise ValueError(f"{field_name} is required and cannot be empty")
        return v


settings = Settings()

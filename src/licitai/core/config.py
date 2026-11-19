from functools import lru_cache

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "LicitAI"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://licitai:licitai_secret@127.0.0.1:5433/licitai_db"  # type: ignore[assignment]
    )

    pncp_base_url: str = "https://pncp.gov.br/api/consulta/v1"
    pncp_timeout_seconds: float = Field(default=30.0, gt=0.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Configuración central de CARDIOAI.

Todos los valores sensibles se leen de variables de entorno (.env).
Los umbrales de riesgo incluidos son de DEMOSTRACIÓN y no constituyen
criterios clínicos validados.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
DEMO_DIR = DATA_DIR / "demo"
DOCS_DIR = BASE_DIR / "docs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "CARDIOAI"
    app_version: str = "1.0.0"
    environment: str = "development"
    api_prefix: str = "/api"

    secret_key: str = "dev-only-change-me-cardioai-2026"
    access_token_minutes: int = 480
    refresh_token_days: int = 7
    jwt_algorithm: str = "HS256"

    database_url: str = f"sqlite:///{(DATA_DIR / 'cardioai.db').as_posix()}"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000"

    admin_email: str = "admin@cardioai.local"
    admin_password: str = "Admin2026!"
    seed_demo_data: bool = True

    # Umbrales de estratificación SOLO para la demo. Configurables por entorno.
    # Deben validarse con datos reales antes de cualquier uso asistencial.
    risk_low_max: float = 0.10
    risk_high_min: float = 0.25
    prediction_horizon_years: int = 10
    ml_trials: int = 8
    ml_cv_folds: int = 3
    demo_seed: int = 42

    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = f"file:///{(DATA_DIR / 'mlruns').as_posix()}"
    llm_provider: str = "none"  # none | openai (opcional, requiere credenciales)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    for d in (DATA_DIR, MODEL_DIR, REPORT_DIR, DEMO_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return Settings()


settings = get_settings()

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized app configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "AtmoGuard API"
    APP_ENV: str = "development"  # development | test | production
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # --- CORS ---
    # Comma-separated list of allowed origins, e.g. "http://localhost:5173,http://localhost:3000"
    CORS_ORIGINS: str = "http://localhost:5173"

    # --- Database ---
    DATABASE_URL: str = "postgresql+psycopg2://localhost:5432/atmoguard"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Weather providers ---
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    HTTP_TIMEOUT_SECONDS: float = 10.0
    HTTP_RETRY_ATTEMPTS: int = 2
    HTTP_RETRY_BACKOFF_SECONDS: float = 0.8
    WEATHER_CACHE_TTL_SECONDS: int = 300  # 5 minutes — matches the frontend's own cache

    # IMD's API is documented (api.imd.gov.in) but gated behind an API key that
    # requires registration/approval — see app/services/providers/imd.py. Unset
    # (None) means "not configured"; the IMD provider stays disabled until a
    # real key is obtained and set here.
    IMD_API_BASE_URL: str = "https://api.imd.gov.in/api/v1"
    IMD_API_KEY: str | None = None

    # --- Automated alert pipeline (Phase 5) ---
    # A single asyncio background loop started in app/main.py's lifespan, not
    # a separate worker process — fine for one `uvicorn` process (same caveat
    # as the in-memory weather cache: not shared/coordinated across
    # `--workers N > 1`). Disabled automatically when APP_ENV == "test" (see
    # main.py) so the test suite never starts a real, repeating Open-Meteo
    # polling loop.
    PIPELINE_ENABLED: bool = True
    PIPELINE_INTERVAL_SECONDS: int = 900  # 15 minutes

    # --- ML model warmup (Phase 2 Step 5) ---
    # Eagerly loads all 12 prediction artifacts at startup instead of on each
    # one's first request (app/ml/model_registry.py:warm_up). Same
    # test-suite exception as PIPELINE_ENABLED -- disabled when
    # APP_ENV == "test" (see main.py) so the test suite doesn't pay the
    # one-time ~1.5s/artifact cold-load cost on every TestClient startup.
    MODEL_WARMUP_ENABLED: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

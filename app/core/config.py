"""
Server configuration loaded from environment variables or a .env file.

Precedence (highest to lowest):
  1. Environment variables   (e.g. ``set DAYLIGHT_PORT=9000``)
  2. ``.env`` file in the project root
  3. Defaults defined below

Create a ``.env`` file from ``.env.example`` to customise settings without
touching code.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    """
    Bind address.
      - "127.0.0.1"  -- localhost only (default, safe)
      - "0.0.0.0"    -- open to the local network
    """

    port: int = 8000
    """Port the server listens on."""

    reload: bool = False
    """Enable auto-reload on code changes (development convenience)."""

    log_level: str = "info"
    """Uvicorn log level (debug, info, warning, error, critical)."""

    model_config = {
        "env_prefix": "DAYLIGHT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()

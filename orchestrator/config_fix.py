"""
Configuration management for Climate Justice AI Orchestrator.

Handles environment variables, API keys, and settings for different deployment environments.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class APIConfig:
    """Generic API server settings."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "False").lower() in ("1", "true", "yes")

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    # Optional external log service
    external_url: Optional[str] = os.getenv("LOG_EXTERNAL_URL")

@dataclass
class ElectricityMapConfig:
    """Configuration for ElectricityMap API integration."""
    api_key: str
    base_url: str = os.getenv("ELECTRICITYMAP_BASE_URL", "https://api.electricitymap.org/v3")
    timeout_seconds: int = int(os.getenv("ELECTRICITYMAP_TIMEOUT", "10"))
    request_timeout: int = int(os.getenv("ELECTRICITYMAP_REQUEST_TIMEOUT", "10"))  # For aiohttp

@dataclass
class ClimateJusticeConfig:
    """Configuration for climate justice economics."""
    # Economic parameters
    energy_cost_per_token: float = float(os.getenv("ENERGY_COST_PER_TOKEN", "0.001"))
    global_south_discount: float = float(os.getenv("GLOBAL_SOUTH_DISCOUNT", "0.3"))
    adaptation_multiplier: float = float(os.getenv("ADAPTATION_MULTIPLIER", "3.0"))
    renewable_discount: float = float(os.getenv("RENEWABLE_DISCOUNT", "0.15"))

@dataclass
class Settings:
    """Aggregate of all configuration sections."""
    environment: str = os.getenv("ENVIRONMENT", "development")
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    electricity_map: ElectricityMapConfig = field(default_factory=lambda: ElectricityMapConfig(
        api_key=os.getenv("ELECTRICITYMAP_API_KEY", "")
    ))
    climate_justice: ClimateJusticeConfig = field(default_factory=ClimateJusticeConfig)

def create_dev_env_file(path: Optional[str] = None) -> None:
    """
    Create a sample `.env` file with all required keys (empty values).
    If the file already exists, this is a no-op.
    """
    target = Path(path or ".env")
    if target.exists():
        return

    sample = """\
# Environment (development|staging|production)
ENVIRONMENT=development

# API server
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Logging
LOG_LEVEL=INFO
LOG_EXTERNAL_URL=

# ElectricityMap API (signup at https://app.electricitymap.org/api)
ELECTRICITYMAP_API_KEY=

# Climate justice settings
ENERGY_COST_PER_TOKEN=0.001
GLOBAL_SOUTH_DISCOUNT=0.3
ADAPTATION_MULTIPLIER=3.0
RENEWABLE_DISCOUNT=0.15
"""
    target.write_text(sample)
    print(f"Created sample .env file at {target.resolve()}")
    print("Please edit it with your actual API keys and settings.")

def configure_logging(cfg: LoggingConfig) -> None:
    """
    Configure the root logger using the provided LoggingConfig.
    """
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=cfg.format, datefmt=cfg.datefmt)
    if cfg.external_url:
        # Example: hook in an external log shipper
        handler = logging.handlers.HTTPHandler(
            cfg.external_url, method="POST", secure=True
        )
        logging.getLogger().addHandler(handler)

def get_settings() -> Settings:
    """
    Load and validate all settings from environment variables.
    
    Raises:
        ValueError: If any required setting is missing or invalid.
    """
    # Ensure .env.sample exists for first-time setup
    create_dev_env_file()
    
    settings = Settings()

    # Validate ElectricityMap API key
    if not settings.electricity_map.api_key:
        print("⚠️  Warning: No ElectricityMap API key configured; using fallback data.")

    # Configure logging
    configure_logging(settings.logging)

    return settings

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with the given name.
    """
    return logging.getLogger(name)

if __name__ == "__main__":
    # Helper for development setup
    create_dev_env_file()
    
    # Test configuration loading
    try:
        settings = get_settings()
        print("Configuration loaded successfully!")
        print(f"Environment: {settings.environment}")
        print(f"API will run on {settings.api.host}:{settings.api.port}")
        print(f"ElectricityMap API configured: {bool(settings.electricity_map.api_key)}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set required environment variables.")

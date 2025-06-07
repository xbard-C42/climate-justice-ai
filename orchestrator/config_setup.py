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
    # dotenv is optional - production might use container env vars
    pass


@dataclass
class ElectricityMapFallbackDefaults:
    """Fallback defaults for different region types."""
    intensity: float
    renewable_ratio: float

@dataclass  # ✅ FIXED: Added missing @dataclass decorator
class ElectricityMapConfig:
    """Configuration for ElectricityMap API integration."""
    api_key: str
    base_url: str = "https://api.electricitymap.org/v3"
    cache_ttl_seconds: int = 300  # 5 minutes
    timeout_seconds: int = 10
    request_timeout: int = 10  # For aiohttp timeout
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 60
    
    # Fallback defaults by region type
    fallback_defaults: Dict[str, ElectricityMapFallbackDefaults] = field(default_factory=lambda: {
        "global_south": ElectricityMapFallbackDefaults(intensity=300, renewable_ratio=0.4),
        "global_north": ElectricityMapFallbackDefaults(intensity=600, renewable_ratio=0.25)
    })
    
    # Legacy fallback regions (for compatibility)
    fallback_regions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "US": {"carbon_intensity": 600, "renewable_percentage": 0.25},
        "EU": {"carbon_intensity": 400, "renewable_percentage": 0.35},
        "BD": {"carbon_intensity": 300, "renewable_percentage": 0.4},
        "KE": {"carbon_intensity": 300, "renewable_percentage": 0.4},
        "CN": {"carbon_intensity": 700, "renewable_percentage": 0.15},
        "IN": {"carbon_intensity": 650, "renewable_percentage": 0.20},
        "BR": {"carbon_intensity": 350, "renewable_percentage": 0.45},
    })


@dataclass  # ✅ FIXED: Added missing @dataclass decorator
class ClimateJusticeConfig:
    """Configuration for climate justice economics."""
    
    # Economic parameters
    energy_cost_per_token: float = 0.001  # Base energy cost per token
    carbon_price_per_gram: float = 0.0001  # Carbon pricing per gram CO2e
    
    # Justice adjustments
    justice_discount_rate: float = 0.3  # 30% discount for Global South benefit
    global_north_premium_rate: float = 0.2  # 20% premium for Global North consumption
    renewable_discount_rate: float = 0.15  # 15% discount for renewable energy
    
    # Priority multipliers
    climate_adaptation_multiplier: float = 3.0  # 3x priority for climate adaptation
    disaster_response_multiplier: float = 4.0   # 4x priority for disaster response
    sustainable_agriculture_multiplier: float = 2.0  # 2x priority for sustainable agriculture
    
    # Global South regions (ISO country codes)
    global_south_regions: set = field(default_factory=lambda: {
        'AF', 'DZ', 'AO', 'AR', 'BD', 'BJ', 'BO', 'BR', 'BF', 'BI', 'KH', 'CM', 
        'CF', 'TD', 'CL', 'CN', 'CO', 'KM', 'CG', 'CD', 'CR', 'CI', 'CU', 'DJ',
        'DO', 'EC', 'EG', 'SV', 'GQ', 'ER', 'ET', 'FJ', 'GA', 'GM', 'GH', 'GT',
        'GN', 'GW', 'GY', 'HT', 'HN', 'IN', 'ID', 'IR', 'IQ', 'JM', 'JO', 'KE',
        'KI', 'KP', 'KR', 'KW', 'LA', 'LB', 'LS', 'LR', 'LY', 'MG', 'MW', 'MY',
        'MV', 'ML', 'MR', 'MU', 'MX', 'FM', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP',
        'NI', 'NE', 'NG', 'NU', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH',
        'QA', 'RW', 'WS', 'ST', 'SA', 'SN', 'SC', 'SL', 'SB', 'SO', 'ZA', 'SS',
        'LK', 'SD', 'SR', 'SZ', 'SY', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TO', 'TT',
        'TN', 'TM', 'TV', 'UG', 'AE', 'UY', 'UZ', 'VU', 'VE', 'VN', 'YE', 'ZM', 'ZW'
    })
    
    # Climate adaptation purposes
    climate_adaptation_purposes: set = field(default_factory=lambda: {
        'climate_adaptation', 'disaster_response', 'drought_prediction', 
        'flood_management', 'wildfire_prediction', 'sea_level_monitoring',
        'agricultural_adaptation', 'water_resource_management', 'climate_refugee_support'
    })
    
    # Renewable energy sources for calculation
    renewable_sources: set = field(default_factory=lambda: {
        'wind', 'solar', 'hydro', 'geothermal', 'biomass'
        # Note: Nuclear is low-carbon but not renewable in most definitions
    })


@dataclass 
class DisasterMonitoringConfig:
    """Configuration for disaster monitoring (GDACS, etc.)."""
    gdacs_api_url: str = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/MAP"
    update_interval_minutes: int = 5
    emergency_threshold_level: int = 3  # GDACS alert level (1-4, where 4 is most severe)
    

@dataclass
class APIConfig:
    """Configuration for FastAPI service."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    
    # CORS settings
    cors_origins: list = field(default_factory=lambda: ["*"])  # Configure for production
    cors_methods: list = field(default_factory=lambda: ["*"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    

@dataclass
class LoggingConfig:
    """Configuration for application logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log to file in production
    log_file: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class ClimateJusticeSettings:
    """Main configuration class combining all settings."""
    
    # Core configurations
    electricity_map: ElectricityMapConfig
    climate_justice: ClimateJusticeConfig
    disaster_monitoring: DisasterMonitoringConfig
    api: APIConfig
    logging: LoggingConfig
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "ClimateJusticeSettings":
        """Create configuration from environment variables."""
        
        # Required environment variables
        electricity_map_api_key = os.getenv("ELECTRICITYMAP_API_KEY")
        if not electricity_map_api_key:
            raise ValueError(
                "ELECTRICITYMAP_API_KEY environment variable is required. "
                "Sign up at https://www.electricitymap.org/api"
            )
        
        # Environment settings
        environment = os.getenv("ENVIRONMENT", "development")
        debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        
        # ElectricityMap configuration
        electricity_map = ElectricityMapConfig(
            api_key=electricity_map_api_key,
            base_url=os.getenv("ELECTRICITYMAP_BASE_URL", "https://api.electricitymap.org/v3"),
            cache_ttl_seconds=int(os.getenv("ELECTRICITYMAP_CACHE_TTL", "300")),
            timeout_seconds=int(os.getenv("ELECTRICITYMAP_TIMEOUT", "10")),
            max_retries=int(os.getenv("ELECTRICITYMAP_MAX_RETRIES", "3")),
            requests_per_minute=int(os.getenv("ELECTRICITYMAP_RATE_LIMIT", "60"))
        )
        
        # Climate justice economics
        climate_justice = ClimateJusticeConfig(
            energy_cost_per_token=float(os.getenv("ENERGY_COST_PER_TOKEN", "0.001")),
            carbon_price_per_gram=float(os.getenv("CARBON_PRICE_PER_GRAM", "0.0001")),
            justice_discount_rate=float(os.getenv("JUSTICE_DISCOUNT_RATE", "0.3")),
            global_north_premium_rate=float(os.getenv("GLOBAL_NORTH_PREMIUM_RATE", "0.2")),
            renewable_discount_rate=float(os.getenv("RENEWABLE_DISCOUNT_RATE", "0.15"))
        )
        
        # Disaster monitoring
        disaster_monitoring = DisasterMonitoringConfig(
            gdacs_api_url=os.getenv("GDACS_API_URL", "https://www.gdacs.org/gdacsapi/api/events/geteventlist/MAP"),
            update_interval_minutes=int(os.getenv("DISASTER_UPDATE_INTERVAL", "5")),
            emergency_threshold_level=int(os.getenv("EMERGENCY_THRESHOLD_LEVEL", "3"))
        )
        
        # API configuration
        api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            log_level=os.getenv("API_LOG_LEVEL", "info"),
            rate_limit_requests_per_minute=int(os.getenv("API_RATE_LIMIT", "100"))
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),  # None by default
            max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE_MB", "100")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
        
        return cls(
            electricity_map=electricity_map,
            climate_justice=climate_justice,
            disaster_monitoring=disaster_monitoring,
            api=api,
            logging=logging_config,
            environment=environment,
            debug=debug
        )
    
    def setup_logging(self):
        """Configure application logging based on settings."""
        import logging.handlers
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.logging.level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.logging.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.log_file,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
settings: Optional[ClimateJusticeSettings] = None

def get_settings() -> ClimateJusticeSettings:
    """Get global settings instance (singleton pattern)."""
    global settings
    if settings is None:
        settings = ClimateJusticeSettings.from_env()
        settings.setup_logging()
    return settings


def get_logger(name: str) -> logging.Logger:
    """Get logger with consistent configuration."""
    # Ensure settings are loaded
    get_settings()
    return logging.getLogger(name)


# Development helper
def create_dev_env_file():
    """Create a sample .env file for development."""
    env_content = """# Climate Justice AI Configuration
# Copy this to .env and fill in your actual values

# Required: ElectricityMap API Key
# Sign up at https://www.electricitymap.org/api
ELECTRICITYMAP_API_KEY=your_api_key_here

# Environment settings
ENVIRONMENT=development
DEBUG=true

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=debug

# Logging
LOG_LEVEL=DEBUG
# LOG_FILE=logs/climate_justice.log  # Uncomment to enable file logging

# Climate justice economics (defaults are usually fine)
# ENERGY_COST_PER_TOKEN=0.001
# CARBON_PRICE_PER_GRAM=0.0001
# JUSTICE_DISCOUNT_RATE=0.3
# GLOBAL_NORTH_PREMIUM_RATE=0.2
# RENEWABLE_DISCOUNT_RATE=0.15

# ElectricityMap settings (defaults are usually fine)
# ELECTRICITYMAP_CACHE_TTL=300
# ELECTRICITYMAP_TIMEOUT=10
# ELECTRICITYMAP_MAX_RETRIES=3
# ELECTRICITYMAP_RATE_LIMIT=60

# Disaster monitoring
# DISASTER_UPDATE_INTERVAL=5
# EMERGENCY_THRESHOLD_LEVEL=3
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print(f"Created sample .env file at {env_file.absolute()}")
        print("Please edit it with your actual API keys and settings.")
    else:
        print(f".env file already exists at {env_file.absolute()}")


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

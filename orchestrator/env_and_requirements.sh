# .env.example - Copy to .env and configure with your values

# Required: ElectricityMap API Key
# Sign up at https://www.electricitymap.org/api
ELECTRICITYMAP_API_KEY=

# Environment settings
ENVIRONMENT=development
DEBUG=true

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=debug
API_RATE_LIMIT=100

# Logging
LOG_LEVEL=DEBUG
# LOG_FILE=logs/climate_justice.log  # Uncomment to enable file logging

# Climate justice economics (defaults usually work)
ENERGY_COST_PER_TOKEN=0.001
CARBON_PRICE_PER_GRAM=0.0001
JUSTICE_DISCOUNT_RATE=0.3
GLOBAL_NORTH_PREMIUM_RATE=0.2
RENEWABLE_DISCOUNT_RATE=0.15

# ElectricityMap settings
ELECTRICITYMAP_CACHE_TTL=300
ELECTRICITYMAP_TIMEOUT=10
ELECTRICITYMAP_MAX_RETRIES=3
ELECTRICITYMAP_RATE_LIMIT=60

# Disaster monitoring
DISASTER_UPDATE_INTERVAL=5
EMERGENCY_THRESHOLD_LEVEL=3

---

# requirements.txt
# Climate Justice AI Orchestrator Dependencies

# Web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# HTTP client for external APIs
aiohttp==3.9.1
httpx==0.25.2

# Data processing
pydantic==2.5.1
pandas==2.1.4

# Configuration management
python-dotenv==1.0.0

# Background tasks and caching
celery==5.3.4
redis==5.0.1

# Database (for future state storage)
sqlalchemy==2.0.23
alembic==1.13.1

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0
httpx==0.25.2  # For testing HTTP clients

# Development tools
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# Monitoring and logging
structlog==23.2.0
prometheus-client==0.19.0

# Security
cryptography==41.0.8

# Optional: Vector database for memory storage
# qdrant-client==1.7.0
# chromadb==0.4.18

# Optional: LLM clients (uncomment as needed)
# openai==1.3.9
# anthropic==0.7.8
# langchain==0.0.339
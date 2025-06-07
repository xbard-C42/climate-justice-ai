from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from orchestrator.config_setup import get_settings, get_logger
from orchestrator.climate_data_implementation import LiveCarbonData
from orchestrator.climate_budget_implementation import CarbonAwareTokenBudget
from orchestrator.types_and_errors import AIRequest, AIResponse

# Initialize settings and logger
default_settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title="Climate Justice AI Orchestrator",
    description="Carbon-aware AI coordination engine with climate justice economics.",
    version="1.0.0",
)

# Allow all CORS origins (customize for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": default_settings.environment}

@app.post("/impact", response_model=AIResponse)
def compute_impact(
    region: str = Query(..., description="ISO country code, e.g. 'BD', 'US'"),
    purpose: str = Query(..., description="Purpose category, e.g. 'climate_adaptation', 'crypto_mining'"),
    tokens: int = Query(1000, ge=1, description="Number of tokens to consume"),
):
    """
    Compute climate impact, true cost, and carbon footprint for a given AI request.
    """
    # Validate request model
    try:
        req = AIRequest(region=region, purpose=purpose, tokens=tokens)
    except Exception as e:
        logger.error(f"Invalid request parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch live carbon data (with fallback)
    carbon_data = LiveCarbonData(default_settings)
    ci, renewable_pct = carbon_data.get_latest_intensity(req.region)

    # Compute budget
    budget = CarbonAwareTokenBudget(default_settings)
    budget.apply_region_and_purpose(req.region, req.purpose)
    budget.actual_usage = req.tokens
    budget.update_climate_metrics(renewable_pct, ci)
    result = budget.calculate_true_cost()

    # Build response
    response = AIResponse(
        region=req.region,
        purpose=req.purpose,
        tokens=req.tokens,
        carbon_footprint=result.carbon_footprint,
        true_cost=result.true_cost,
        impact_score=result.climate_impact_score,
    )
    return response

"""
Climate Justice AI - FastAPI Application
Revolutionary climate justice economics through AI resource allocation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any
import asyncio

from .config_setup import get_settings, get_logger
from .types_and_errors import AIRequest, AIResponse, ClimateImpactData
from .climate_budget_implementation import CarbonAwareTokenBudget, create_test_budget
from .climate_data_implementation import LiveCarbonData

# Initialize
settings = get_settings()
logger = get_logger()

app = FastAPI(
    title="Climate Justice AI",
    description="Revolutionary AI system prioritizing climate justice through economic incentives",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
carbon_data = LiveCarbonData()
budget_system = create_test_budget()

@app.on_event("startup")
async def startup_event():
    """Initialize the climate justice system"""
    logger.info("🌍 Climate Justice AI Starting Up!")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"🔌 ElectricityMap API: {'Configured' if settings.electricitymap_api_key else 'Fallback Mode'}")
    logger.info("⚡ Revolutionary climate justice economics activated!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "climate_justice": "active",
        "revolution": "in_progress"
    }

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "message": "🌍 Climate Justice AI - Revolutionary Economics",
        "description": "AI system that automatically prioritizes Global South climate adaptation",
        "endpoints": {
            "health": "/health",
            "process": "/v1/ai/process",
            "climate_data": "/v1/climate/data/{region}"
        }
    }

@app.post("/v1/ai/process")
async def process_ai_request(request_data: Dict[str, Any]):
    """
    Process AI request with climate justice economics
    
    Automatically applies:
    - Global South economic benefits
    - Climate adaptation priority multipliers
    - Carbon-aware resource allocation
    """
    try:
        # Extract request data
        req = request_data.get("request", {})
        prompt = req.get("prompt", "")
        region = req.get("region", "US")
        purpose = req.get("purpose", "general")
        
        logger.info(f"🌍 Processing request: {purpose} in {region}")
        
        # Get climate data for region
        climate_info = await carbon_data.get_carbon_data(region)
        
        # Calculate climate impact using budget system
        climate_impact = await budget_system.calculate_climate_impact(
            region=region,
            purpose=purpose,
            carbon_intensity=climate_info["carbon_intensity"],
            renewable_percentage=climate_info["renewable_percentage"]
        )
        
        # Simulate AI response (in real system, this would call actual AI)
        if purpose == "climate_adaptation":
            ai_response = f"🌊 Climate adaptation analysis for {region}: Implementing flood prediction and early warning systems using satellite imagery and local weather patterns. Priority given to vulnerable communities."
        elif purpose == "renewable_energy":
            ai_response = f"🌱 Renewable energy optimization for {region}: Solar and wind potential analysis with community-based implementation strategies."
        else:
            ai_response = f"General AI response for {region}: Standard processing with carbon-aware resource allocation."
        
        # Log the revolutionary economics
        logger.info(f"💰 Impact Score: {climate_impact['impact_score']}")
        logger.info(f"🌍 Global South Benefit: ${climate_impact['global_south_benefit']}")
        logger.info(f"⚡ Carbon Footprint: {climate_impact['carbon_footprint']} gCO2e")
        
        return {
            "response": {
                "ai_response": ai_response,
                "climate_impact": climate_impact,
                "region": region,
                "purpose": purpose,
                "revolution_status": "active"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Climate justice system error: {str(e)}")

@app.get("/v1/climate/data/{region}")
async def get_climate_data(region: str):
    """Get climate data for a specific region"""
    try:
        data = await carbon_data.get_carbon_data(region)
        return {
            "region": region,
            "climate_data": data,
            "source": "ElectricityMap API" if settings.electricitymap_api_key else "Fallback Data"
        }
    except Exception as e:
        logger.error(f"❌ Error fetching climate data for {region}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Climate data error: {str(e)}")

@app.get("/v1/justice/economics")
async def get_justice_economics():
    """Get current climate justice economic parameters"""
    return {
        "justice_discount_rate": settings.justice_discount_rate,
        "global_north_premium": settings.global_north_premium,
        "renewable_discount": settings.renewable_discount,
        "global_south_regions": len(settings.global_south_regions),
        "climate_purposes": len(settings.climate_purposes),
        "revolution_level": "maximum"
    }

# For running directly
if __name__ == "__main__":
    uvicorn.run(
        "orchestrator.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
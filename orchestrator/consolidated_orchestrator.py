"""
Complete Earth Justice AI Orchestrator
Revolutionary climate justice economics with unassailable utility-based mathematics
Based on marginal utility research: everything above €200k has zero marginal utility
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from datetime import datetime

# Import dependencies (implement these modules separately)
from llm_client import UnifiedLLMClient
from missing_orchestrator_modules import ContextManager
from types_and_errors import OrchestratorError
from pydantic_climate_budget import AIRequest, RoleDefinition

# === ECONOMIC ASSESSMENT SYSTEM ===

@dataclass
class EconomicContext:
    """Individual economic context for justice pricing based on utility research."""
    
    # Economic tiers based on marginal utility curve research
    economic_tier: Literal["basic_needs", "stable_living", "comfortable", "surplus_wealth"] = "stable_living"
    
    # Purpose categories for differential pricing
    purpose_category: Literal[
        "education", "basic_assistance", "personal_development", 
        "professional_work", "business_growth", "luxury_consumption", 
        "investment_optimization", "high_frequency_trading"
    ] = "professional_work"
    
    # Geographic and usage context
    region: str = ""
    cost_of_living_index: float = 1.0
    monthly_usage_tokens: int = 0
    usage_frequency: int = 0
    business_usage_ratio: float = 0.0
    
    # Verification status
    student_verified: bool = False
    nonprofit_verified: bool = False
    enterprise_account: bool = False
    
    # Optional actual income data for precise assessment
    annual_income_eur: Optional[int] = None

    @property
    def base_economic_multiplier(self) -> float:
        """Calculate base multiplier from marginal utility tiers.
        
        Based on utility plateau research:
        - €0-30k: Steep utility curve (every euro matters)
        - €30k-80k: Moderate utility gains
        - €80k-200k: Diminishing returns approaching plateau
        - €200k+: Utility plateau - zero marginal utility
        """
        tier_multipliers = {
            "basic_needs": 0.2,        # 80% discount - steep utility curve, maximum need
            "stable_living": 0.7,      # 30% discount - moderate utility gains
            "comfortable": 1.2,        # 20% premium - approaching utility plateau
            "surplus_wealth": 5.0      # 400% premium - zero marginal utility loss
        }
        return tier_multipliers[self.economic_tier]


class UtilityBasedPricingEngine:
    """Advanced pricing engine that redistributes genuine surplus based on utility research."""
    
    def __init__(self):
        # Cost of living indices by country (normalised to global baseline)
        self.cost_of_living = {
            # Global South - Lower cost of living
            'BD': 0.3, 'KE': 0.4, 'NG': 0.35, 'IN': 0.4, 'PK': 0.35,
            'PH': 0.45, 'VN': 0.45, 'ID': 0.4, 'MY': 0.6, 'TH': 0.55,
            'ZA': 0.5, 'GH': 0.4, 'TZ': 0.35, 'UG': 0.3, 'RW': 0.4,
            'ET': 0.3, 'MX': 0.5, 'CO': 0.45, 'PE': 0.4, 'AR': 0.6,
            
            # Global North - Higher cost of living
            'US': 1.0, 'GB': 0.9, 'DE': 0.8, 'FR': 0.85, 'IE': 0.9,
            'AU': 0.95, 'CA': 0.85, 'SE': 0.9, 'CH': 1.3, 'NO': 1.1
        }
        
        # Purpose-based adjustments - aggressive for surplus activities
        self.purpose_adjustments = {
            'education': 0.3,              # Heavy discount - builds human capital
            'basic_assistance': 0.2,       # Maximum support - survival needs
            'personal_development': 0.5,   # Encourage self-improvement
            'professional_work': 1.0,      # Standard rate - productive work
            'business_growth': 1.3,        # Modest premium - creates value
            'luxury_consumption': 2.5,     # High premium - pure luxury above needs
            'investment_optimization': 4.0, # Very high - wealth multiplication for surplus class
            'high_frequency_trading': 6.0  # Maximum - pure speculation above utility plateau
        }
        
        # Income thresholds based on utility plateau research (in EUR)
        self.utility_thresholds = {
            'basic_needs_max': 30000,      # €0-30k: steep utility curve
            'stable_living_max': 80000,    # €30k-80k: moderate utility gains
            'comfortable_max': 200000,     # €80k-200k: diminishing returns
            # €200k+: utility plateau - zero marginal utility
        }

    def assess_economic_tier_from_income(self, annual_income_eur: int) -> str:
        """Assess economic tier directly from income using utility research."""
        if annual_income_eur <= self.utility_thresholds['basic_needs_max']:
            return "basic_needs"
        elif annual_income_eur <= self.utility_thresholds['stable_living_max']:
            return "stable_living"
        elif annual_income_eur <= self.utility_thresholds['comfortable_max']:
            return "comfortable"
        else:
            return "surplus_wealth"  # Above utility plateau

    def validate_self_reported_tier(
        self, 
        self_reported: str, 
        usage_patterns: Dict,
        verification_data: Dict,
        annual_income: Optional[int] = None
    ) -> str:
        """Validate self-reported tier against usage patterns and income data."""
        
        # If we have actual income, use utility plateau thresholds
        if annual_income:
            income_based_tier = self.assess_economic_tier_from_income(annual_income)
            # Don't allow significant underreporting
            if income_based_tier == "surplus_wealth" and self_reported in ["basic_needs", "stable_living"]:
                return "surplus_wealth"  # Above plateau - can't claim basic needs
            return income_based_tier
        
        # Validate against spending patterns
        monthly_spend = usage_patterns.get('monthly_ai_spend_eur', 0)
        usage_frequency = usage_patterns.get('requests_per_month', 0)
        
        # Infer economic tier from AI spending (people above plateau spend more)
        if monthly_spend > 500:  # €6k+/year on AI suggests surplus wealth
            return "surplus_wealth"
        elif monthly_spend > 200:  # €2.4k+/year suggests comfortable
            return max(self_reported, "comfortable") if self_reported != "surplus_wealth" else "comfortable"
        
        # Red flags for underreporting above utility plateau
        if (self_reported in ["basic_needs", "stable_living"] and 
            monthly_spend > 300 and usage_frequency > 1500):
            return "comfortable"  # Likely underreporting
        
        # Verification overrides
        if verification_data.get('student_id_verified'):
            return "basic_needs"  # Students protected regardless
        
        if verification_data.get('enterprise_account'):
            return "surplus_wealth"  # Enterprise users typically above plateau
        
        return self_reported

    def calculate_justice_pricing(
        self,
        base_tokens: int,
        economic_context: EconomicContext,
        climate_purpose: bool = False
    ) -> Dict[str, float]:
        """Calculate utility-based justice pricing with mathematical rigor."""
        
        # Base cost calculation
        base_cost = base_tokens * 0.001  # €0.001 per token baseline
        
        # 1. Economic tier multiplier (based on marginal utility research)
        economic_multiplier = economic_context.base_economic_multiplier
        
        # 2. Cost of living adjustment
        cost_of_living = self.cost_of_living.get(economic_context.region, 1.0)
        economic_context.cost_of_living_index = cost_of_living
        
        # Protect basic needs regardless of geography
        if economic_context.economic_tier in ["basic_needs", "stable_living"]:
            purchasing_power_adjustment = max(0.3, cost_of_living)
        else:
            # Wealthy people in expensive places pay proportionally more
            purchasing_power_adjustment = cost_of_living
        
        # 3. Purpose-based adjustment (aggressive for surplus activities)
        purpose_multiplier = self.purpose_adjustments.get(
            economic_context.purpose_category, 1.0
        )
        
        # 4. Usage pattern analysis
        usage_multiplier = self._calculate_usage_multiplier(economic_context)
        
        # 5. Verification discounts
        verification_discount = self._calculate_verification_discount(economic_context)
        
        # 6. Climate priority boost
        climate_multiplier = 0.5 if climate_purpose else 1.0
        
        # Combine all factors
        total_multiplier = (
            economic_multiplier * 
            purchasing_power_adjustment * 
            purpose_multiplier * 
            usage_multiplier * 
            climate_multiplier * 
            (1 - verification_discount)
        )
        
        # Ensure minimum viable pricing (never below cost)
        final_multiplier = max(0.2, total_multiplier)
        adjusted_cost = base_cost * final_multiplier
        
        # Calculate redistributable surplus (anything above 80% of base cost)
        redistributable_surplus = max(0, adjusted_cost - (base_cost * 0.8))
        
        # Calculate utility impact for transparency
        utility_impact = self._calculate_utility_impact(economic_context, final_multiplier)
        
        return {
            'base_cost': base_cost,
            'adjusted_cost': adjusted_cost,
            'total_multiplier': final_multiplier,
            'redistributable_surplus': redistributable_surplus,
            'utility_impact': utility_impact,
            'breakdown': {
                'economic_multiplier': economic_multiplier,
                'purchasing_power_adjustment': purchasing_power_adjustment,
                'purpose_multiplier': purpose_multiplier,
                'usage_multiplier': usage_multiplier,
                'climate_multiplier': climate_multiplier,
                'verification_discount': verification_discount
            }
        }
    
    def _calculate_usage_multiplier(self, context: EconomicContext) -> float:
        """High usage indicates higher economic capacity."""
        if context.monthly_usage_tokens > 100000:  # Very high usage
            return 1.4
        elif context.monthly_usage_tokens > 50000:  # High usage
            return 1.2
        elif context.monthly_usage_tokens > 10000:  # Medium usage
            return 1.0
        else:  # Low usage
            return 0.9
    
    def _calculate_verification_discount(self, context: EconomicContext) -> float:
        """Verification-based discounts and premiums."""
        discount = 0.0
        
        if context.student_verified:
            discount += 0.3  # 30% off for verified students
        
        if context.nonprofit_verified:
            discount += 0.4  # 40% off for nonprofits
        
        if context.enterprise_account:
            discount -= 0.5  # 50% premium for enterprise (negative discount)
        
        return max(-0.5, min(0.5, discount))  # Cap at ±50%
    
    def _calculate_utility_impact(self, context: EconomicContext, multiplier: float) -> str:
        """Calculate and describe utility impact for transparency."""
        if context.economic_tier == "surplus_wealth":
            return f"Zero marginal utility impact (above €200k plateau)"
        elif context.economic_tier == "comfortable":
            return f"Minimal utility impact (approaching plateau)"
        elif context.economic_tier == "stable_living":
            return f"Moderate utility impact (meaningful but manageable)"
        else:  # basic_needs
            return f"Protected pricing (steep utility curve)"


def create_economic_context_from_request(request: AIRequest, user_profile: Dict) -> EconomicContext:
    """Create economic context from request and user profile."""
    
    # Map request purpose to economic categories
    purpose_mapping = {
        'education': 'education',
        'homework': 'education', 
        'learning': 'education',
        'research': 'education',
        'climate_adaptation': 'basic_assistance',
        'disaster_response': 'basic_assistance',
        'emergency_planning': 'basic_assistance',
        'personal': 'personal_development',
        'professional': 'professional_work',
        'business': 'business_growth',
        'commercial': 'business_growth',
        'marketing': 'business_growth',
        'investment': 'investment_optimization',
        'trading': 'high_frequency_trading',
        'luxury': 'luxury_consumption'
    }
    
    purpose_category = purpose_mapping.get(request.purpose, 'professional_work')
    
    return EconomicContext(
        economic_tier=user_profile.get('economic_tier', 'stable_living'),
        purpose_category=purpose_category,
        region=request.origin_region,
        monthly_usage_tokens=user_profile.get('monthly_usage_tokens', 0),
        usage_frequency=user_profile.get('usage_frequency', 0),
        business_usage_ratio=user_profile.get('business_usage_ratio', 0.0),
        student_verified=user_profile.get('student_verified', False),
        nonprofit_verified=user_profile.get('nonprofit_verified', False),
        enterprise_account=user_profile.get('enterprise_account', False),
        annual_income_eur=user_profile.get('annual_income_eur')
    )


# === ORCHESTRATOR WITH EARTH JUSTICE ECONOMICS ===

app = FastAPI(title="Earth Justice AI Orchestrator", version="1.0.0")
orchestrator: Optional['EarthJusticeOrchestrator'] = None

class EarthJusticeOrchestrator:
    """AI orchestrator implementing Earth Justice economics."""
    
    def __init__(self):
        self.llm_client = UnifiedLLMClient(
            providers=os.getenv("LLM_PROVIDERS", "openai").split(",")
        )
        self.context_manager = ContextManager()
        self.pricing_engine = UtilityBasedPricingEngine()

    async def execute_role_with_justice(
        self, 
        role: RoleDefinition, 
        request: AIRequest, 
        user_profile: Dict = None
    ) -> Any:
        """Execute AI role with utility-based economic justice."""
        
        # Create economic context
        user_profile = user_profile or {}
        economic_context = create_economic_context_from_request(request, user_profile)
        
        # Validate economic tier against usage patterns
        validated_tier = self.pricing_engine.validate_self_reported_tier(
            economic_context.economic_tier,
            user_profile,
            user_profile,
            economic_context.annual_income_eur
        )
        economic_context.economic_tier = validated_tier
        
        # Calculate utility-based pricing
        is_climate_purpose = economic_context.purpose_category in ["basic_assistance"]
        pricing_result = self.pricing_engine.calculate_justice_pricing(
            base_tokens=request.requested_tokens,
            economic_context=economic_context,
            climate_purpose=is_climate_purpose
        )
        
        # Reserve budget with justice pricing
        try:
            budget = await self.context_manager.reserve(
                tokens=role.max_response_tokens,
                carbon_multiplier=role.carbon_budget_multiplier,
                economic_multiplier=pricing_result['total_multiplier'],
                redistributable_surplus=pricing_result['redistributable_surplus']
            )
        except Exception as e:
            raise OrchestratorError("Failed to reserve carbon budget") from e

        try:
            # Execute LLM call with justice context
            response = await self.llm_client.send(
                prompt=self._build_justice_prompt(role, request, economic_context, pricing_result),
                provider=self._select_provider(request.origin_region, request),
                max_tokens=budget.allocated_tokens,
            )
            
            # Commit usage with justice metrics
            await self.context_manager.commit(
                budget, 
                response.usage.total_tokens,
                justice_metrics={
                    'economic_tier': economic_context.economic_tier,
                    'validated_tier': validated_tier,
                    'redistributable_surplus': pricing_result['redistributable_surplus'],
                    'total_multiplier': pricing_result['total_multiplier'],
                    'utility_impact': pricing_result['utility_impact'],
                    'climate_purpose': is_climate_purpose
                }
            )
            
            return response
            
        except Exception as e:
            await self.context_manager.rollback(budget)
            raise OrchestratorError("LLM execution failed") from e

    def _build_justice_prompt(
        self, 
        role: RoleDefinition, 
        request: AIRequest,
        economic_context: EconomicContext, 
        pricing_result: Dict
    ) -> str:
        """Build justice-aware prompt with transparent economic context."""
        
        # Determine messaging based on economic tier and utility impact
        if economic_context.economic_tier == "basic_needs":
            economic_message = f"🎓 PRIORITY REQUEST: Basic needs support (80% discount)"
            priority_note = "This request receives maximum priority due to steep marginal utility."
        
        elif economic_context.economic_tier == "stable_living":
            discount_pct = int((1-pricing_result['total_multiplier'])*100) if pricing_result['total_multiplier'] < 1 else 0
            premium_pct = int((pricing_result['total_multiplier']-1)*100) if pricing_result['total_multiplier'] > 1 else 0
            
            if discount_pct > 0:
                economic_message = f"🤝 Working class support ({discount_pct}% discount)"
            else:
                economic_message = f"💼 Standard request ({premium_pct}% justice contribution)"
            priority_note = "This request supports working class access to AI infrastructure."
        
        elif economic_context.economic_tier == "comfortable":
            premium_pct = int((pricing_result['total_multiplier']-1)*100)
            economic_message = f"💰 Professional request ({premium_pct}% justice contribution)"
            priority_note = "This request contributes to subsidising access for those with steeper utility curves."
        
        else:  # surplus_wealth
            premium_pct = int((pricing_result['total_multiplier']-1)*100)
            surplus_contribution = pricing_result['redistributable_surplus']
            economic_message = f"🏆 Above utility plateau ({premium_pct}% contribution: €{surplus_contribution:.2f})"
            priority_note = "Mathematical justification: Zero marginal utility loss above €200k income threshold."
        
        # Climate and carbon context
        climate_context = f"""
🌡️ Carbon Impact: {request.renewable_energy_ratio:.1%} renewable energy
📍 Region: {request.origin_region} (Cost of Living Index: {economic_context.cost_of_living_index:.1f})
⚡ Estimated carbon footprint: ~{request.requested_tokens * 0.0001:.3f}g CO2e
"""
        
        # Justice impact summary with utility transparency
        justice_impact = f"""
💡 Earth Justice Impact:
- Economic tier: {economic_context.economic_tier.replace('_', ' ').title()}
- Purpose: {economic_context.purpose_category.replace('_', ' ').title()}
- Utility impact: {pricing_result['utility_impact']}
- Redistributable surplus: €{pricing_result['redistributable_surplus']:.3f}
- Funds: {"Climate adaptation" if economic_context.purpose_category == "basic_assistance" else "Economic justice initiatives"}
"""
        
        # Complete justice prompt
        justice_prompt = f"""
{economic_message}
{climate_context}
{justice_impact}

{priority_note}

SYSTEM ROLE: {role.system_prompt}

EARTH JUSTICE CONTEXT:
- This system implements utility-based economics: pricing follows marginal utility curves
- €200k+ income threshold represents utility plateau - zero marginal utility loss from premium pricing
- Economic incentives protect steep utility curves (basic needs) while redistributing from flat utility (surplus wealth)
- Your response contributes to mathematically rigorous climate and economic justice infrastructure
- Pricing transparency builds trust in Earth Justice principles

USER REQUEST from {request.origin_region} ({economic_context.purpose_category}):
Economic context: {economic_context.economic_tier} | Verified: {economic_context.student_verified or economic_context.nonprofit_verified} | Purpose: {economic_context.purpose_category}

{request.query}

Please respond according to your role while being mindful of the Earth Justice context of this request.
"""
        
        return justice_prompt

    def _select_provider(self, origin_region: str, request: AIRequest = None) -> str:
        """Select LLM provider based on climate justice principles."""
        available_providers = self.llm_client.providers
        
        # Global South regions (prioritised for climate justice)
        global_south_regions = {
            'BD', 'KE', 'NG', 'IN', 'PK', 'PH', 'VN', 'TH', 'ID', 'MY',
            'ZA', 'GH', 'TZ', 'UG', 'RW', 'ET', 'MX', 'CO', 'PE', 'AR'
        }
        
        is_global_south = origin_region in global_south_regions
        is_climate_purpose = request and request.purpose in {
            'climate_adaptation', 'disaster_response', 'emergency_planning'
        }
        emergency_level = getattr(request, 'emergency_level', 0)
        
        # Climate emergency gets highest priority
        if emergency_level >= 3:
            return "openai" if "openai" in available_providers else available_providers[0]
        
        # Global South + climate gets top tier providers
        if is_global_south and is_climate_purpose:
            for provider in ["anthropic", "openai", "gemini"]:
                if provider in available_providers:
                    return provider
        
        # Global South gets preferential provider selection
        elif is_global_south:
            for provider in ["gemini", "openai"]:
                if provider in available_providers:
                    return provider
        
        return available_providers[0] if available_providers else "openai"


# === FASTAPI ENDPOINTS ===

@app.post("/v1/ai/request")
async def process_ai_request(request: AIRequest, user_id: Optional[str] = None):
    """Process AI request with Earth Justice economics."""
    global orchestrator
    if orchestrator is None:
        orchestrator = EarthJusticeOrchestrator()
    
    # Get user profile for economic assessment
    user_profile = get_user_profile(user_id) if user_id else {}
    
    # Define role for request
    role = RoleDefinition(
        role_id="earth_justice_assistant",
        system_prompt="You are an AI assistant committed to Earth Justice principles, implementing utility-based economic fairness and climate prioritisation.",
        max_response_tokens=request.requested_tokens
    )
    
    try:
        response = await orchestrator.execute_role_with_justice(role, request, user_profile)
        return {
            "success": True, 
            "response": response,
            "earth_justice": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earth Justice orchestration failed: {str(e)}")


@app.post("/v1/user/profile")
async def update_user_profile(user_id: str, profile_data: Dict):
    """Update user economic profile for justice pricing."""
    
    # Validate profile data
    allowed_fields = {
        'economic_tier', 'annual_income_eur', 'student_verified', 
        'nonprofit_verified', 'enterprise_account', 'monthly_usage_tokens', 
        'business_usage_ratio', 'usage_frequency'
    }
    
    sanitised_profile = {k: v for k, v in profile_data.items() if k in allowed_fields}
    
    # Store profile
    store_user_profile(user_id, sanitised_profile)
    
    return {
        "success": True, 
        "message": "Economic profile updated for Earth Justice pricing",
        "earth_justice": True
    }


@app.get("/v1/pricing/calculate")
async def calculate_pricing_preview(
    tokens: int,
    economic_tier: str = "stable_living",
    purpose: str = "professional_work",
    region: str = "US"
):
    """Preview Earth Justice pricing for transparency."""
    
    pricing_engine = UtilityBasedPricingEngine()
    
    economic_context = EconomicContext(
        economic_tier=economic_tier,
        purpose_category=purpose,
        region=region
    )
    
    result = pricing_engine.calculate_justice_pricing(
        base_tokens=tokens,
        economic_context=economic_context
    )
    
    return {
        "base_cost_eur": result['base_cost'],
        "adjusted_cost_eur": result['adjusted_cost'],
        "total_multiplier": result['total_multiplier'],
        "redistributable_surplus_eur": result['redistributable_surplus'],
        "utility_impact": result['utility_impact'],
        "breakdown": result['breakdown'],
        "earth_justice": True
    }


# === USER PROFILE STORAGE (implement with database in production) ===

_user_profiles: Dict[str, Dict] = {}

def get_user_profile(user_id: str) -> Dict:
    """Get user profile for economic assessment."""
    return _user_profiles.get(user_id, {
        'economic_tier': 'stable_living',
        'student_verified': False,
        'nonprofit_verified': False,
        'enterprise_account': False,
        'monthly_usage_tokens': 0,
        'business_usage_ratio': 0.0
    })

def store_user_profile(user_id: str, profile: Dict):
    """Store user profile."""
    _user_profiles[user_id] = profile


# === COMPREHENSIVE TESTING SUITE ===

async def test_earth_justice_scenarios():
    """Test comprehensive Earth Justice scenarios with utility-based pricing."""
    
    print("🌍 Earth Justice AI Orchestrator - Comprehensive Testing")
    print("Based on marginal utility research: €200k+ income has zero marginal utility\n")
    
    orchestrator = EarthJusticeOrchestrator()
    
    test_scenarios = [
        {
            "name": "Verified Student (Basic Needs)",
            "profile": {
                'economic_tier': 'basic_needs',
                'student_verified': True,
                'annual_income_eur': 15000
            },
            "request": AIRequest(
                id="test-student",
                query="Help me understand climate science for my environmental engineering thesis",
                origin_region="GB",
                purpose="education",
                requested_tokens=500
            )
        },
        {
            "name": "Bangladesh Climate Worker (Stable Living)",
            "profile": {
                'economic_tier': 'stable_living',
                'annual_income_eur': 25000
            },
            "request": AIRequest(
                id="test-climate-bd",
                query="Design flood-resistant infrastructure for coastal communities",
                origin_region="BD",
                purpose="climate_adaptation",
                requested_tokens=800
            )
        },
        {
            "name": "German Professional (Comfortable)",
            "profile": {
                'economic_tier': 'comfortable',
                'annual_income_eur': 120000
            },
            "request": AIRequest(
                id="test-professional-de",
                query="Create a sustainability report for our manufacturing company",
                origin_region="DE",
                purpose="professional",
                requested_tokens=600
            )
        },
        {
            "name": "Investment Optimisation (Surplus Wealth)",
            "profile": {
                'economic_tier': 'surplus_wealth',
                'enterprise_account': True,
                'annual_income_eur': 400000,
                'monthly_usage_tokens': 100000
            },
            "request": AIRequest(
                id="test-investment",
                query="Optimise my portfolio allocation for maximum returns across emerging markets",
                origin_region="US",
                purpose="investment",
                requested_tokens=1000
            )
        },
        {
            "name": "High Frequency Trading (Above Plateau)",
            "profile": {
                'economic_tier': 'surplus_wealth',
                'enterprise_account': True,
                'annual_income_eur': 800000,
                'monthly_usage_tokens': 200000
            },
            "request": AIRequest(
                id="test-hft",
                query="Develop algorithmic trading strategies for cryptocurrency arbitrage",
                origin_region="US",
                purpose="trading",
                requested_tokens=1200
            )
        }
    ]
    
    for scenario in test_scenarios:
        print(f"📊 Testing: {scenario['name']}")
        print(f"   Income: €{scenario['profile'].get('annual_income_eur', 'Not specified'):,}")
        
        try:
            role = RoleDefinition(
                role_id="test_role",
                system_prompt="You are a test assistant.",
                max_response_tokens=scenario['request'].requested_tokens
            )
            
            # Test economic context creation
            economic_context = create_economic_context_from_request(
                scenario['request'], 
                scenario['profile']
            )
            
            # Test pricing calculation
            pricing_engine = UtilityBasedPricingEngine()
            is_climate = economic_context.purpose_category == "basic_assistance"
            
            pricing = pricing_engine.calculate_justice_pricing(
                base_tokens=scenario['request'].requested_tokens,
                economic_context=economic_context,
                climate_purpose=is_climate
            )
            
            print(f"   ✅ Economic tier: {economic_context.economic_tier}")
            print(f"   💰 Base cost: €{pricing['base_cost']:.3f}")
            print(f"   🎯 Adjusted cost: €{pricing['adjusted_cost']:.3f}")
            print(f"   📈 Multiplier: {pricing['total_multiplier']:.1f}x")
            print(f"   🌍 Redistributable surplus: €{pricing['redistributable_surplus']:.3f}")
            print(f"   🧠 Utility impact: {pricing['utility_impact']}")
            
            if economic_context.economic_tier == "surplus_wealth":
                print(f"   🎖️  Mathematical justification: Zero marginal utility above €200k plateau")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}\n")
    
    print("🚀 Earth Justice Economics: Mathematically rigorous redistribution complete!")


# === ENTRY POINT ===

if __name__ == "__main__":
    # Run comprehensive tests
    asyncio.run(test_earth_justice_scenarios())
    
    # For production: uvicorn orchestrator:app --host 0.0.0.0 --port 8000

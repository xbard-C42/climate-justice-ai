def _select_provider(self, origin_region: str, purpose: str = "", emergency_level: int = 0) -> str:
    """
    Select LLM provider based on climate justice principles.
    
    Priority Logic:
    - Global South: Prefer local/open models, then cheapest APIs
    - Climate emergencies: Use most reliable providers
    - Global North non-climate: Standard commercial APIs
    """
    available_providers = self.llm_client.providers
    
    # Global South regions get priority access to efficient models
    global_south_regions = {
        'BD', 'KE', 'NG', 'IN', 'PK', 'PH', 'VN', 'TH', 'ID', 'MY',
        'ZA', 'GH', 'TZ', 'UG', 'RW', 'ET', 'MX', 'CO', 'PE', 'AR'
    }
    
    # Climate adaptation/emergency purposes
    climate_purposes = {'climate_adaptation', 'disaster_response', 'emergency_planning'}
    
    if emergency_level >= 3:  # Climate emergency - use most reliable
        if "openai" in available_providers:
            return "openai"  # Most reliable for emergencies
        elif "anthropic" in available_providers:
            return "anthropic"
    
    if origin_region in global_south_regions:
        # Global South priority: local models first, then efficient commercial
        if purpose in climate_purposes:
            # Climate work gets best available
            if "anthropic" in available_providers:
                return "anthropic"  # Often most capable for complex reasoning
            elif "openai" in available_providers:
                return "openai"
        else:
            # Non-climate Global South work: prefer efficient/local
            if "local" in available_providers:
                return "local"  # Prefer local models for cost efficiency
            elif "gemini" in available_providers:
                return "gemini"  # Often cheaper than OpenAI/Anthropic
            elif "openai" in available_providers:
                return "openai"
    
    else:  # Global North regions
        if purpose in climate_purposes:
            # Climate work from Global North - use best available
            if "anthropic" in available_providers:
                return "anthropic"
            elif "openai" in available_providers:
                return "openai"
        else:
            # Non-climate Global North work - standard commercial
            if "openai" in available_providers:
                return "openai"
            elif "gemini" in available_providers:
                return "gemini"
    
    # Fallback to first available
    return available_providers[0] if available_providers else "openai"


def _build_justice_prompt(self, role: RoleDefinition, request: AIRequest) -> str:
    """
    Build justice-aware prompt incorporating economic context and priorities.
    """
    # Calculate justice context (this would normally come from your pricing engine)
    global_south_regions = {
        'BD', 'KE', 'NG', 'IN', 'PK', 'PH', 'VN', 'TH', 'ID', 'MY',
        'ZA', 'GH', 'TZ', 'UG', 'RW', 'ET', 'MX', 'CO', 'PE', 'AR'
    }
    
    is_global_south = request.origin_region in global_south_regions
    is_climate_purpose = request.purpose in {'climate_adaptation', 'disaster_response', 'emergency_planning'}
    
    # Economic context
    if is_global_south and is_climate_purpose:
        economic_context = "🌍 PRIORITY REQUEST: Global South climate adaptation (70% discount applied)"
        priority_note = "This request receives maximum priority due to climate justice principles."
    elif is_global_south:
        economic_context = "🌍 Global South request (30% discount applied)"
        priority_note = "This request supports Global South development."
    elif is_climate_purpose:
        economic_context = "🌱 Climate adaptation request (3x priority multiplier)"
        priority_note = "This request supports critical climate adaptation work."
    else:
        economic_context = "💼 Standard commercial request"
        priority_note = "Standard processing with carbon offset included."
    
    # Carbon context
    carbon_context = f"""
🌡️ Carbon Impact: {request.renewable_energy_ratio:.1%} renewable energy
📍 Region: {request.origin_region}
⚡ Estimated carbon footprint: ~{request.requested_tokens * 0.0001:.3f}g CO2e
"""
    
    # Build the complete prompt
    justice_prompt = f"""
{economic_context}
{carbon_context}

{priority_note}

SYSTEM ROLE: {role.system_prompt}

JUSTICE CONTEXT:
- This AI system automatically prioritises climate adaptation and Global South requests
- Economic incentives align with climate justice through differential pricing
- Your response contributes to climate-positive AI infrastructure

USER REQUEST from {request.origin_region} ({request.purpose}):
{request.query}

Please respond according to your role while being mindful of the climate justice context of this request.
"""
    
    return justice_prompt


def _calculate_justice_metrics(self, request: AIRequest, response_tokens: int) -> dict:
    """
    Calculate justice impact metrics for this request.
    """
    global_south_regions = {
        'BD', 'KE', 'NG', 'IN', 'PK', 'PH', 'VN', 'TH', 'ID', 'MY',
        'ZA', 'GH', 'TZ', 'UG', 'RW', 'ET', 'MX', 'CO', 'PE', 'AR'
    }
    
    is_global_south = request.origin_region in global_south_regions
    is_climate_purpose = request.purpose in {'climate_adaptation', 'disaster_response', 'emergency_planning'}
    
    # Base cost calculation (simplified)
    base_cost = response_tokens * 0.001  # $0.001 per token baseline
    
    # Justice adjustments
    discount_multiplier = 1.0
    if is_global_south and is_climate_purpose:
        discount_multiplier = 0.3  # 70% discount
    elif is_global_south:
        discount_multiplier = 0.7  # 30% discount
    elif is_climate_purpose:
        discount_multiplier = 0.8  # 20% discount for climate work
    else:
        discount_multiplier = 1.2  # 20% premium for Global North non-climate
    
    adjusted_cost = base_cost * discount_multiplier
    
    # Climate impact
    carbon_per_token = 0.0001  # gCO2e per token (simplified)
    carbon_footprint = response_tokens * carbon_per_token * (1 - request.renewable_energy_ratio)
    
    # Justice benefit calculation
    global_south_benefit = max(0, base_cost - adjusted_cost) if is_global_south else 0
    climate_adaptation_value = response_tokens * 0.002 if is_climate_purpose else 0
    
    return {
        'base_cost': base_cost,
        'adjusted_cost': adjusted_cost,
        'carbon_footprint': carbon_footprint,
        'global_south_benefit': global_south_benefit,
        'climate_adaptation_value': climate_adaptation_value,
        'justice_priority_score': (
            (2.0 if is_global_south else 1.0) * 
            (3.0 if is_climate_purpose else 1.0) *
            (1.5 if request.emergency_level > 0 else 1.0)
        )
    }
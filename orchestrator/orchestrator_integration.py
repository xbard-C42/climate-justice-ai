# Updated orchestrator.py with sophisticated economic assessment

def _build_justice_prompt(self, role: RoleDefinition, request: AIRequest, economic_context: EconomicContext, pricing_result: Dict) -> str:
    """Build justice-aware prompt with nuanced economic context."""
    
    # Determine priority level and messaging
    if economic_context.economic_tier == "basic_needs":
        if economic_context.purpose_category in ["education", "basic_assistance"]:
            economic_message = f"🎓 PRIORITY REQUEST: {economic_context.purpose_category} for basic needs (70% discount applied)"
            priority_note = "This request receives maximum priority due to economic justice principles."
        else:
            economic_message = f"🌍 Basic needs support ({int((1-pricing_result['total_multiplier'])*100)}% discount applied)"
            priority_note = "This request supports someone with basic economic needs."
    
    elif economic_context.economic_tier == "stable_living":
        discount_pct = int((1-pricing_result['total_multiplier'])*100) if pricing_result['total_multiplier'] < 1 else 0
        premium_pct = int((pricing_result['total_multiplier']-1)*100) if pricing_result['total_multiplier'] > 1 else 0
        
        if discount_pct > 0:
            economic_message = f"🤝 Working class support ({discount_pct}% discount applied)"
        elif premium_pct > 0:
            economic_message = f"💼 Standard request ({premium_pct}% contribution to justice fund)"
        else:
            economic_message = "💼 Standard rate request"
        priority_note = "This request supports working class access to AI."
    
    elif economic_context.economic_tier == "comfortable":
        premium_pct = int((pricing_result['total_multiplier']-1)*100)
        economic_message = f"💰 Professional request ({premium_pct}% contribution to justice fund)"
        priority_note = "This request contributes to subsidizing access for those with fewer resources."
    
    else:  # surplus_wealth
        premium_pct = int((pricing_result['total_multiplier']-1)*100)
        surplus_contribution = pricing_result['redistributable_surplus']
        economic_message = f"🏆 High-capacity request ({premium_pct}% justice contribution: ${surplus_contribution:.2f})"
        priority_note = "This request significantly funds climate and economic justice initiatives."
    
    # Climate context
    climate_context = f"""
🌡️ Carbon Impact: {request.renewable_energy_ratio:.1%} renewable energy
📍 Region: {request.origin_region} (Cost of Living Index: {economic_context.cost_of_living_index:.1f})
⚡ Estimated carbon footprint: ~{request.requested_tokens * 0.0001:.3f}g CO2e
"""
    
    # Justice impact summary
    justice_impact = f"""
💡 Justice Impact Summary:
- Economic tier: {economic_context.economic_tier.replace('_', ' ').title()}
- Purpose: {economic_context.purpose_category.replace('_', ' ').title()}
- Redistributable surplus: ${pricing_result['redistributable_surplus']:.3f}
- Supports: {"Climate adaptation" if economic_context.purpose_category == "basic_assistance" else "Economic justice"}
"""
    
    # Build the complete prompt
    justice_prompt = f"""
{economic_message}
{climate_context}
{justice_impact}

{priority_note}

SYSTEM ROLE: {role.system_prompt}

JUSTICE CONTEXT:
- This AI system uses sophisticated economic assessment to redistribute actual surplus wealth
- Economic incentives protect basic needs while asking those with surplus to contribute more
- Your response contributes to climate and economic justice infrastructure
- Pricing is transparent and based on genuine economic capacity, not crude regional assumptions

USER REQUEST from {request.origin_region} ({economic_context.purpose_category}):
Economic context: {economic_context.economic_tier} | Verified: {economic_context.student_verified or economic_context.nonprofit_verified}

{request.query}

Please respond according to your role while being mindful of the economic justice context of this request.
"""
    
    return justice_prompt


async def execute_role_with_justice(self, role: RoleDefinition, request: AIRequest, user_profile: Dict = None) -> Any:
    """Execute AI role with sophisticated economic assessment."""
    
    # Create economic context from request and user profile
    user_profile = user_profile or {}
    economic_context = create_economic_context_from_request(request, user_profile)
    
    # Calculate sophisticated pricing
    pricing_engine = SophisticatedPricingEngine()
    is_climate_purpose = economic_context.purpose_category in ["basic_assistance", "climate_adaptation"]
    
    pricing_result = pricing_engine.calculate_justice_pricing(
        base_tokens=request.requested_tokens,
        economic_context=economic_context,
        climate_purpose=is_climate_purpose
    )
    
    # Reserve carbon budget with sophisticated pricing
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
        # LLM call with sophisticated justice context
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
                'redistributable_surplus': pricing_result['redistributable_surplus'],
                'total_multiplier': pricing_result['total_multiplier'],
                'climate_purpose': is_climate_purpose
            }
        )
        
        return response
        
    except Exception as e:
        await self.context_manager.rollback(budget)
        raise OrchestratorError("LLM execution failed") from e


# Updated FastAPI endpoint with user profiles
@app.post("/v1/ai/request")
async def process_ai_request(request: AIRequest, user_id: Optional[str] = None):
    """Process AI request with sophisticated economic assessment."""
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator()
    
    # Get user profile (in production, this would come from database)
    user_profile = get_user_profile(user_id) if user_id else {}
    
    role = RoleDefinition(
        role_id="general_assistant",
        system_prompt="You are a helpful AI assistant committed to climate and economic justice.",
        max_response_tokens=request.requested_tokens
    )
    
    try:
        response = await orchestrator.execute_role_with_justice(role, request, user_profile)
        return {"success": True, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# User profile management endpoint
@app.post("/v1/user/profile")
async def update_user_profile(user_id: str, profile_data: Dict):
    """Update user economic profile for justice pricing."""
    
    # Validate and sanitize profile data
    allowed_fields = {
        'economic_tier', 'student_verified', 'nonprofit_verified', 
        'enterprise_account', 'monthly_usage_tokens', 'business_usage_ratio'
    }
    
    sanitized_profile = {k: v for k, v in profile_data.items() if k in allowed_fields}
    
    # Store profile (in production: database)
    store_user_profile(user_id, sanitized_profile)
    
    return {"success": True, "message": "Profile updated for justice pricing"}


def get_user_profile(user_id: str) -> Dict:
    """Get user profile for economic assessment."""
    # In production: database lookup
    # For now: return default profile
    return {
        'economic_tier': 'stable_living',
        'student_verified': False,
        'nonprofit_verified': False,
        'enterprise_account': False,
        'monthly_usage_tokens': 0,
        'business_usage_ratio': 0.0
    }


def store_user_profile(user_id: str, profile: Dict):
    """Store user profile."""
    # In production: database storage
    pass


# Updated smoke tests with economic context
async def test_sophisticated_pricing():
    """Test sophisticated economic assessment."""
    
    orchestrator = Orchestrator()
    
    # Test case 1: Verified student doing homework
    student_profile = {
        'economic_tier': 'basic_needs',
        'student_verified': True,
        'monthly_usage_tokens': 2000
    }
    
    student_request = AIRequest(
        id="test-student",
        query="Help me understand climate change for my environmental science class",
        origin_region="US",
        purpose="education",
        requested_tokens=300
    )
    
    print("🎓 Testing verified student pricing...")
    try:
        response = await orchestrator.execute_role_with_justice(
            RoleDefinition(role_id="tutor", system_prompt="You are an educational tutor.", max_response_tokens=300),
            student_request,
            student_profile
        )
        print("✅ Student request successful - should have heavy discount")
        print("Response includes education priority:", "🎓 PRIORITY REQUEST" in str(response))
    except Exception as e:
        print(f"❌ Student test failed: {e}")
    
    # Test case 2: Enterprise high-frequency trading
    executive_profile = {
        'economic_tier': 'surplus_wealth',
        'enterprise_account': True,
        'monthly_usage_tokens': 200000,
        'business_usage_ratio': 1.0
    }
    
    trading_request = AIRequest(
        id="test-trading",
        query="Optimize my algorithmic trading strategy for maximum profit",
        origin_region="US", 
        purpose="trading",
        requested_tokens=1000
    )
    
    print("\n💰 Testing high-capacity trading pricing...")
    try:
        response = await orchestrator.execute_role_with_justice(
            RoleDefinition(role_id="analyst", system_prompt="You are a financial analyst.", max_response_tokens=1000),
            trading_request,
            executive_profile
        )
        print("✅ Trading request successful - should have high premium")
        print("Response includes premium:", "🏆 High-capacity request" in str(response))
    except Exception as e:
        print(f"❌ Trading test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_sophisticated_pricing())

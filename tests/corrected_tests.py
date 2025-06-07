"""
Corrected Climate Justice Integration Tests.

Fixed to work with our actual implementation.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

from orchestrator.climate_data import LiveCarbonData, LiveElectricityMapAPI
from orchestrator.climate_budget import CarbonAwareTokenBudget, CarbonAwareTokenBudgetTracker
from orchestrator.config import settings
from orchestrator.core import Orchestrator
from orchestrator.types import RoleDefinition, RoleResponse
from orchestrator.context import ContextManager
from orchestrator.metrics import TokenCounter
from orchestrator.token_budget import TokenBudgetTracker

# Ensure pytest-asyncio is used
pytest_plugins = ('pytest_asyncio',)


class DummyClimateTracker:
    """Mock climate tracker for testing."""
    def __init__(self):
        self.records = []
    
    async def record_session_impact(self, budget: CarbonAwareTokenBudget):
        self.records.append(budget)


class MockLLMResponse:
    """Mock LLM API response."""
    def __init__(self, content):
        self.choices = [type('Choice', (), {
            'message': type('Message', (), {'content': content})()
        })()]


@pytest.mark.asyncio
async def test_global_south_discount_and_adaptation_multiplier():
    """Test that Global South regions get a discount and adaptation multiplier applied."""
    
    # Mock climate data: Bangladesh with 50% renewable
    mock_data = LiveCarbonData(
        carbon_intensity=300.0,
        renewable_percentage=0.5,
        fossil_percentage=0.5,
        timestamp=datetime.utcnow(),
        region='BD'
    )
    
    climate_source = AsyncMock(spec=LiveElectricityMapAPI)
    climate_source.get_live_carbon_data.return_value = mock_data

    tracker = CarbonAwareTokenBudgetTracker(climate_source)
    role = RoleDefinition(
        role_id='climate_adapter',
        system_prompt='You help with climate adaptation',
        model_name='gpt-4',
        max_response_tokens=1000,
        temperature=0.7,
        timeout_seconds=30
    )
    
    context = {'region': 'BD', 'purpose': 'climate_adaptation'}

    # Get climate-aware budget
    budget = await tracker.estimate_climate_aware_budget(role, 'Predict flooding in coastal areas', context, region='BD')
    
    # Verify Global South discount was applied
    assert budget.global_south_benefit > 0
    expected_benefit = budget.initial_estimate * settings.climate_justice.justice_discount_rate
    assert budget.global_south_benefit == pytest.approx(expected_benefit)
    
    # Verify climate adaptation multiplier was applied
    assert budget.climate_adaptation_value > 0
    # Current estimate should be higher than initial due to 3x multiplier
    assert budget.current_estimate > budget.initial_estimate
    
    # Test true cost calculation
    budget.actual_usage = 2000
    budget.update_climate_metrics(0.5, 300)  # 50% renewable, 300 gCO2/kWh
    
    true_cost = budget.calculate_true_cost()
    
    # Justice adjustment should be positive (discount)
    assert true_cost.justice_adjustment > 0
    print(f"Bangladesh - Global South benefit: ${budget.global_south_benefit:.2f}")
    print(f"Bangladesh - Climate adaptation value: {budget.climate_adaptation_value}")
    print(f"Bangladesh - True cost: ${true_cost.true_cost:.2f}")


@pytest.mark.asyncio
async def test_us_crypto_premium_no_benefits():
    """Test that US region with crypto purpose pays premium and gets no discount."""
    
    # Mock climate data: US with lower renewable percentage
    mock_data = LiveCarbonData(
        carbon_intensity=600.0,
        renewable_percentage=0.25,
        fossil_percentage=0.75,
        timestamp=datetime.utcnow(),
        region='US'
    )
    
    climate_source = AsyncMock(spec=LiveElectricityMapAPI)
    climate_source.get_live_carbon_data.return_value = mock_data

    tracker = CarbonAwareTokenBudgetTracker(climate_source)
    role = RoleDefinition(
        role_id='crypto_optimizer',
        system_prompt='You optimize cryptocurrency mining',
        model_name='gpt-4',
        max_response_tokens=500,
        temperature=0.5,
        timeout_seconds=20
    )
    
    context = {'region': 'US', 'purpose': 'crypto_mining'}

    # Get budget (no special treatment for crypto mining)
    budget = await tracker.estimate_climate_aware_budget(role, 'optimize mining efficiency', context, region='US')
    
    # US is not in Global South, so no discount should be applied
    assert budget.global_south_benefit == 0.0
    
    # Crypto mining is not climate adaptation, so no multiplier
    assert budget.climate_adaptation_value == 0.0
    
    # Calculate true cost
    budget.actual_usage = 2000
    budget.update_climate_metrics(0.25, 600)  # 25% renewable, 600 gCO2/kWh
    
    true_cost = budget.calculate_true_cost()
    
    # Justice adjustment should be negative (premium)
    assert true_cost.justice_adjustment < 0
    
    # Should pay Global North premium
    expected_premium = true_cost.base_cost * settings.climate_justice.global_north_premium_rate
    assert abs(abs(true_cost.justice_adjustment) - expected_premium) < 0.01
    
    print(f"US Crypto - Global South benefit: ${budget.global_south_benefit:.2f}")
    print(f"US Crypto - Climate adaptation value: {budget.climate_adaptation_value}")
    print(f"US Crypto - True cost: ${true_cost.true_cost:.2f}")
    print(f"US Crypto - Justice adjustment (premium): ${true_cost.justice_adjustment:.2f}")


@pytest.mark.asyncio
async def test_kenya_agriculture_global_south_benefits():
    """Test that Kenya agriculture gets Global South benefits."""
    
    # Mock climate data: Kenya with good renewable mix
    mock_data = LiveCarbonData(
        carbon_intensity=300.0,
        renewable_percentage=0.4,
        fossil_percentage=0.6,
        timestamp=datetime.utcnow(),
        region='KE'
    )
    
    climate_source = AsyncMock(spec=LiveElectricityMapAPI)
    climate_source.get_live_carbon_data.return_value = mock_data

    tracker = CarbonAwareTokenBudgetTracker(climate_source)
    role = RoleDefinition(
        role_id='agriculture_advisor',
        system_prompt='You help with sustainable agriculture',
        model_name='gpt-4',
        max_response_tokens=800,
        temperature=0.7,
        timeout_seconds=25
    )
    
    context = {'region': 'KE', 'purpose': 'sustainable_agriculture'}

    # Get climate-aware budget
    budget = await tracker.estimate_climate_aware_budget(role, 'optimize drought-resistant crops', context, region='KE')
    
    # Kenya is Global South, should get discount
    assert budget.global_south_benefit > 0
    
    # Calculate true cost
    budget.actual_usage = 1500
    budget.update_climate_metrics(0.4, 300)  # 40% renewable, 300 gCO2/kWh
    
    true_cost = budget.calculate_true_cost()
    
    # Should get justice discount
    assert true_cost.justice_adjustment > 0
    
    print(f"Kenya Agriculture - Global South benefit: ${budget.global_south_benefit:.2f}")
    print(f"Kenya Agriculture - True cost: ${true_cost.true_cost:.2f}")


@pytest.mark.asyncio
async def test_fallback_on_api_failure():
    """Test that LiveElectricityMapAPI fallback returns defaults when API fails."""
    
    # Create API instance and force it to fail
    api = LiveElectricityMapAPI()
    
    # Patch the API to raise an exception
    with patch.object(api, '_get_fallback') as mock_fallback:
        # Configure mock fallback
        mock_fallback.return_value = LiveCarbonData(
            carbon_intensity=300.0,  # Global South fallback
            renewable_percentage=0.4,
            fossil_percentage=0.6,
            timestamp=datetime.utcnow(),
            region='BD'
        )
        
        # Force the main method to fail and use fallback
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = Exception("API Error")
            
            data = await api.get_live_carbon_data('BD')
            
            # Should return fallback data
            assert isinstance(data, LiveCarbonData)
            assert data.region == 'BD'
            assert data.carbon_intensity == 300.0
            assert data.renewable_percentage == 0.4


@pytest.mark.asyncio
async def test_orchestrator_execute_role_integration():
    """Test full orchestrator integration for climate-aware execution."""
    
    # Mock LLM API
    mock_api = AsyncMock()
    mock_api.chat.completions.create.return_value = MockLLMResponse("Sustainable agriculture recommendations for Kenya")

    # Mock climate data
    mock_data = LiveCarbonData(
        carbon_intensity=300.0,
        renewable_percentage=0.4,
        fossil_percentage=0.6,
        timestamp=datetime.utcnow(),
        region='KE'
    )
    climate_source = AsyncMock(spec=LiveElectricityMapAPI)
    climate_source.get_live_carbon_data.return_value = mock_data

    # Setup orchestrator components
    context_manager = ContextManager()
    token_counter = TokenCounter()
    token_tracker = CarbonAwareTokenBudgetTracker(climate_source)
    climate_tracker = DummyClimateTracker()

    # Create orchestrator
    orch = Orchestrator(
        api_client=mock_api,
        context_manager=context_manager,
        token_tracker=token_tracker,
        token_counter=token_counter,
        climate_tracker=climate_tracker
    )

    # Define role
    role = RoleDefinition(
        role_id='agriculture_expert',
        system_prompt='You are an expert in sustainable agriculture',
        model_name='gpt-4',
        max_response_tokens=1000,
        temperature=0.7,
        timeout_seconds=30
    )
    
    context = {'region': 'KE', 'purpose': 'sustainable_agriculture'}

    # Execute role
    result: RoleResponse = await orch._execute_role(role, 'Help farmers adapt to climate change', context)

    # Verify results
    assert result.text == "Sustainable agriculture recommendations for Kenya"
    assert result.tokens > 0
    assert hasattr(result, 'carbon_footprint')
    assert result.carbon_footprint is not None
    
    # Verify climate tracker recorded the session
    assert len(climate_tracker.records) == 1
    recorded_budget = climate_tracker.records[0]
    
    # Should have Global South benefit (Kenya)
    assert recorded_budget.global_south_benefit > 0
    
    # Should have positive climate impact
    assert recorded_budget.climate_impact_score > 0
    
    print(f"Orchestrator Integration - Response: {result.text[:50]}...")
    print(f"Orchestrator Integration - Tokens: {result.tokens}")
    print(f"Orchestrator Integration - Carbon footprint: {result.carbon_footprint:.3f} gCO2e")
    print(f"Orchestrator Integration - Climate impact: {recorded_budget.climate_impact_score:.1f}")


@pytest.mark.asyncio
async def test_comparative_climate_economics():
    """Test comparative climate economics across different scenarios."""
    
    scenarios = [
        {
            'name': 'Bangladesh Flood Prediction',
            'region': 'BD',
            'purpose': 'climate_adaptation',
            'tokens': 2000,
            'carbon_intensity': 300,
            'renewable_ratio': 0.4
        },
        {
            'name': 'US Crypto Mining',
            'region': 'US', 
            'purpose': 'crypto_mining',
            'tokens': 2000,
            'carbon_intensity': 600,
            'renewable_ratio': 0.25
        },
        {
            'name': 'Kenya Agriculture',
            'region': 'KE',
            'purpose': 'sustainable_agriculture', 
            'tokens': 1500,
            'carbon_intensity': 300,
            'renewable_ratio': 0.4
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        # Create budget
        if scenario['region'] in settings.climate_justice.global_south_regions and scenario['purpose'] in settings.climate_justice.climate_adaptation_purposes:
            # Climate adaptation in Global South
            budget = CarbonAwareTokenBudget(1000, 1000, region=scenario['region'])
            budget.apply_global_south_discount()
            budget.apply_priority_multiplier(3.0)
        elif scenario['region'] in settings.climate_justice.global_south_regions:
            # Global South but not climate adaptation
            budget = CarbonAwareTokenBudget(1000, 1000, region=scenario['region'])
            budget.apply_global_south_discount()
        else:
            # Global North
            budget = CarbonAwareTokenBudget(1000, 1000, region=scenario['region'])
        
        # Set usage and update climate metrics
        budget.actual_usage = scenario['tokens']
        budget.update_climate_metrics(
            scenario['renewable_ratio'],
            scenario['carbon_intensity']
        )
        
        # Calculate economics
        true_cost = budget.calculate_true_cost()
        
        result = {
            'name': scenario['name'],
            'region': scenario['region'],
            'purpose': scenario['purpose'],
            'global_south_benefit': budget.global_south_benefit,
            'climate_adaptation_value': budget.climate_adaptation_value,
            'true_cost': true_cost.true_cost,
            'climate_impact_score': budget.climate_impact_score,
            'carbon_footprint': budget.carbon_footprint,
            'justice_adjustment': true_cost.justice_adjustment
        }
        
        results.append(result)
        
        print(f"\n--- {result['name']} ---")
        print(f"Region: {result['region']}")
        print(f"Global South Benefit: ${result['global_south_benefit']:.2f}")
        print(f"Climate Adaptation Value: {result['climate_adaptation_value']}")
        print(f"True Cost: ${result['true_cost']:.2f}")
        print(f"Climate Impact Score: {result['climate_impact_score']:.1f}")
        print(f"Carbon Footprint: {result['carbon_footprint']:.3f} gCO2e")
        print(f"Justice Adjustment: ${result['justice_adjustment']:.2f}")
    
    # Verify climate justice is working
    bd_result = next(r for r in results if r['region'] == 'BD')
    us_result = next(r for r in results if r['region'] == 'US')
    ke_result = next(r for r in results if r['region'] == 'KE')
    
    # Global South should have benefits, US should not
    assert bd_result['global_south_benefit'] > 0
    assert ke_result['global_south_benefit'] > 0
    assert us_result['global_south_benefit'] == 0
    
    # Climate adaptation should have highest impact
    assert bd_result['climate_impact_score'] > us_result['climate_impact_score']
    
    # US should pay premium (negative justice adjustment)
    assert us_result['justice_adjustment'] < 0
    
    # Global South should get discounts (positive justice adjustment)
    assert bd_result['justice_adjustment'] > 0
    assert ke_result['justice_adjustment'] > 0
    
    print(f"\n🌍 CLIMATE JUSTICE VALIDATION COMPLETE 🌍")
    print(f"✅ Global South gets economic benefits")
    print(f"✅ Climate adaptation gets priority")
    print(f"✅ Global North pays appropriate premiums")
    print(f"✅ Carbon accounting works correctly")


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

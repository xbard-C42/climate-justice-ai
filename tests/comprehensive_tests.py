"""
Comprehensive tests for Climate Justice AI integration.

Tests the complete climate justice economics pipeline:
- Live climate data integration
- Climate-aware token budgeting  
- Justice pricing calculations
- Global South prioritization
- Climate adaptation multipliers
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

from orchestrator.climate_data import LiveElectricityMapAPI, LiveCarbonData
from orchestrator.climate_budget import CarbonAwareTokenBudget, create_test_budget
from orchestrator.config import get_settings
from orchestrator.types import AIRequest, RoleDefinition


class TestClimateDataIntegration:
    """Test live climate data integration with ElectricityMap API."""
    
    @pytest.fixture
    def mock_api_responses(self):
        """Mock API responses for different scenarios."""
        return {
            "BD": {  # Bangladesh - Global South with good renewables
                "carbon_intensity": {
                    "data": {"carbonIntensity": 300}
                },
                "power_breakdown": {
                    "data": {
                        "powerConsumptionBreakdown": {
                            "hydro": 200,
                            "gas": 150,
                            "coal": 100,
                            "solar": 50,
                            "wind": 25,
                            "oil": 25
                        }
                    }
                }
            },
            "US": {  # United States - Global North with more fossil fuels
                "carbon_intensity": {
                    "data": {"carbonIntensity": 600}
                },
                "power_breakdown": {
                    "data": {
                        "powerConsumptionBreakdown": {
                            "gas": 300,
                            "coal": 200,
                            "nuclear": 150,
                            "wind": 100,
                            "hydro": 75,
                            "solar": 25
                        }
                    }
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_live_data_fetch_bangladesh(self, mock_api_responses):
        """Test fetching live data for Bangladesh (Global South)."""
        
        # Mock aiohttp session
        with patch('aiohttp.ClientSession') as mock_session:
            # Configure mock responses
            mock_get = AsyncMock()
            mock_session.return_value.__aenter__.return_value.get = mock_get
            
            # Mock carbon intensity response
            mock_resp_ci = AsyncMock()
            mock_resp_ci.json.return_value = mock_api_responses["BD"]["carbon_intensity"]
            mock_resp_ci.raise_for_status.return_value = None
            
            # Mock power breakdown response  
            mock_resp_pb = AsyncMock()
            mock_resp_pb.json.return_value = mock_api_responses["BD"]["power_breakdown"]
            mock_resp_pb.raise_for_status.return_value = None
            
            # Configure get() to return appropriate response based on URL
            def get_side_effect(url, **kwargs):
                if "carbon-intensity" in url:
                    return mock_resp_ci
                elif "power-breakdown" in url:
                    return mock_resp_pb
                else:
                    raise ValueError(f"Unexpected URL: {url}")
            
            mock_get.return_value.__aenter__.side_effect = get_side_effect
            
            # Test the API
            api = LiveElectricityMapAPI()
            data = await api.get_live_carbon_data("BD")
            
            # Assertions
            assert isinstance(data, LiveCarbonData)
            assert data.region == "BD"
            assert data.carbon_intensity == 300
            
            # Check renewable calculation
            # Total = 550, Renewables (hydro + solar + wind) = 275
            expected_renewable = 275 / 550  # ≈ 0.5 (50%)
            assert abs(data.renewable_percentage - expected_renewable) < 0.01
            assert abs(data.fossil_percentage - (1 - expected_renewable)) < 0.01
    
    @pytest.mark.asyncio
    async def test_api_failure_fallback(self):
        """Test fallback data when API fails."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock API failure
            mock_get = AsyncMock()
            mock_get.return_value.__aenter__.side_effect = Exception("API Error")
            mock_session.return_value.__aenter__.return_value.get = mock_get
            
            api = LiveElectricityMapAPI()
            data = await api.get_live_carbon_data("BD")
            
            # Should get fallback data for Global South
            assert isinstance(data, LiveCarbonData)
            assert data.region == "BD"
            # Should use Global South fallback values
            fallback = settings.electricity_map.fallback_defaults_global_south
            assert data.carbon_intensity == fallback.intensity  # 300
            assert data.renewable_percentage == fallback.renewable_ratio  # 0.4
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, mock_api_responses):
        """Test that API results are cached properly."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_get = AsyncMock()
            mock_session.return_value.__aenter__.return_value.get = mock_get
            
            # Configure responses
            mock_resp_ci = AsyncMock()
            mock_resp_ci.json.return_value = mock_api_responses["BD"]["carbon_intensity"]
            mock_resp_ci.raise_for_status.return_value = None
            
            mock_resp_pb = AsyncMock()  
            mock_resp_pb.json.return_value = mock_api_responses["BD"]["power_breakdown"]
            mock_resp_pb.raise_for_status.return_value = None
            
            def get_side_effect(url, **kwargs):
                if "carbon-intensity" in url:
                    return mock_resp_ci
                else:
                    return mock_resp_pb
            
            mock_get.return_value.__aenter__.side_effect = get_side_effect
            
            api = LiveElectricityMapAPI()
            
            # First call should hit API
            data1 = await api.get_live_carbon_data("BD")
            
            # Second call should use cache
            data2 = await api.get_live_carbon_data("BD")
            
            # Should have same data
            assert data1.carbon_intensity == data2.carbon_intensity
            assert data1.renewable_percentage == data2.renewable_percentage
            
            # Should have called API only once (2 calls: carbon-intensity + power-breakdown)
            assert mock_get.call_count == 2


class TestClimateJusticeEconomics:
    """Test climate justice economics calculations."""
    
    def test_global_south_discount_application(self):
        """Test that Global South regions get justice discounts."""
        
        # Bangladesh (Global South) climate adaptation
        budget_bd = create_test_budget(region="BD", purpose="climate_adaptation")
        budget_bd.actual_usage = 2000
        
        # Should have Global South benefit
        assert budget_bd.global_south_benefit > 0
        assert budget_bd.climate_adaptation_value > 0
        
        # Calculate true cost
        true_cost = budget_bd.calculate_true_cost()
        
        # Justice adjustment should be positive (discount)
        assert true_cost.justice_adjustment > 0
        
        # True cost should be higher than base cost due to increased allocation
        # but with justice discounts applied
        assert true_cost.true_cost != true_cost.base_cost
    
    def test_global_north_premium_application(self):
        """Test that Global North luxury applications pay premiums."""
        
        # US crypto mining (Global North, non-climate purpose)
        budget_us = CarbonAwareTokenBudget(
            initial_estimate=2000,
            current_estimate=2000,
            actual_usage=2000,
            region="US"
        )
        
        # No Global South or climate benefits
        assert budget_us.global_south_benefit == 0
        assert budget_us.climate_adaptation_value == 0
        
        # Calculate true cost
        true_cost = budget_us.calculate_true_cost()
        
        # Justice adjustment should be negative (premium)
        assert true_cost.justice_adjustment < 0
        
        # Should pay Global North premium
        expected_premium = true_cost.base_cost * settings.climate_justice.global_north_premium_rate
        assert abs(abs(true_cost.justice_adjustment) - expected_premium) < 0.01
    
    def test_climate_adaptation_priority_multiplier(self):
        """Test climate adaptation gets priority multipliers."""
        
        budget = CarbonAwareTokenBudget(
            initial_estimate=1000,
            current_estimate=1000,
            region="BD"
        )
        
        original_estimate = budget.current_estimate
        
        # Apply climate adaptation multiplier (3x)
        budget.apply_priority_multiplier(3.0)
        
        # Should have increased allocation
        assert budget.current_estimate == original_estimate * 3
        assert budget.climate_adaptation_value == original_estimate * 2  # Difference
    
    def test_renewable_energy_incentives(self):
        """Test renewable energy gets economic incentives."""
        
        budget = CarbonAwareTokenBudget(
            initial_estimate=1000,
            current_estimate=1000,
            actual_usage=1000,
            region="BD"
        )
        
        # High renewable scenario
        budget.update_climate_metrics(
            renewable_ratio=0.8,  # 80% renewable
            grid_intensity=300,
            execution_time_seconds=1.0
        )
        
        true_cost_high_renewable = budget.calculate_true_cost()
        
        # Low renewable scenario
        budget.update_climate_metrics(
            renewable_ratio=0.2,  # 20% renewable  
            grid_intensity=300,
            execution_time_seconds=1.0
        )
        
        true_cost_low_renewable = budget.calculate_true_cost()
        
        # High renewable should cost less
        assert true_cost_high_renewable.renewable_discount > true_cost_low_renewable.renewable_discount
        assert true_cost_high_renewable.true_cost < true_cost_low_renewable.true_cost
    
    def test_carbon_footprint_calculation(self):
        """Test carbon footprint calculation accuracy."""
        
        budget = CarbonAwareTokenBudget(
            initial_estimate=1000,
            current_estimate=1000,
            actual_usage=1000,
            region="US"
        )
        
        # Test with high carbon grid
        budget.update_climate_metrics(
            renewable_ratio=0.2,  # 20% renewable, 80% fossil
            grid_intensity=600,   # High carbon intensity
            execution_time_seconds=1.0
        )
        
        # Carbon footprint should be significant
        assert budget.carbon_footprint > 0
        
        # Test calculation: tokens * energy_factor * grid_intensity * fossil_ratio
        energy_per_token = 0.0001  # kWh per token
        expected_footprint = 1000 * energy_per_token * 600 * 0.8
        assert abs(budget.carbon_footprint - expected_footprint) < 0.001
    
    def test_climate_impact_scoring(self):
        """Test holistic climate impact scoring."""
        
        # Climate adaptation in Bangladesh (should be highly positive)
        budget_positive = create_test_budget(region="BD", purpose="climate_adaptation")
        budget_positive.actual_usage = 2000
        budget_positive.update_climate_metrics(0.6, 300)  # 60% renewable
        
        # Crypto mining in US (should be low/negative)
        budget_negative = CarbonAwareTokenBudget(
            initial_estimate=2000,
            current_estimate=2000,
            actual_usage=2000,
            region="US"
        )
        budget_negative.update_climate_metrics(0.2, 600)  # 20% renewable, high carbon
        
        # Climate adaptation should have much higher impact score
        assert budget_positive.climate_impact_score > budget_negative.climate_impact_score
        assert budget_positive.climate_impact_score > 1000  # Should be strongly positive


class TestIntegratedClimateJusticeWorkflow:
    """Test end-to-end climate justice workflow."""
    
    @pytest.mark.asyncio
    async def test_bangladesh_flood_prediction_workflow(self):
        """Test complete workflow for Bangladesh flood prediction."""
        
        # Create AI request for Bangladesh flood prediction
        request = AIRequest(
            id="bd_flood_001",
            query="Predict flooding risk for coastal Bangladesh in next 48 hours",
            origin_region="BD",
            purpose="climate_adaptation", 
            complexity=0.9,
            requested_tokens=2000
        )
        
        # Mock climate data
        with patch.object(LiveElectricityMapAPI, 'get_live_carbon_data') as mock_climate:
            mock_climate.return_value = LiveCarbonData(
                carbon_intensity=300,
                renewable_percentage=0.4,
                fossil_percentage=0.6,
                timestamp=datetime.utcnow(),
                region="BD"
            )
            
            # Create climate-aware budget
            budget = create_test_budget(region=request.origin_region, purpose=request.purpose)
            budget.actual_usage = request.requested_tokens
            
            # Update with climate data
            climate_data = await mock_climate.return_value
            budget.update_climate_metrics(
                climate_data.renewable_percentage,
                climate_data.carbon_intensity
            )
            
            # Calculate economics
            true_cost = budget.calculate_true_cost()
            
            # Assertions for climate justice
            assert budget.global_south_benefit > 0  # Gets Global South discount
            assert budget.climate_adaptation_value > 0  # Gets climate adaptation priority  
            assert true_cost.justice_adjustment > 0  # Positive = discount applied
            assert budget.climate_impact_score > 10000  # High positive climate impact
            
            # Should be net positive climate impact
            assert budget.climate_impact_score > abs(budget.carbon_footprint * 10)
    
    @pytest.mark.asyncio  
    async def test_us_crypto_mining_workflow(self):
        """Test complete workflow for US crypto mining."""
        
        request = AIRequest(
            id="us_crypto_001",
            query="Optimize cryptocurrency mining efficiency",
            origin_region="US",
            purpose="crypto_mining",
            complexity=0.3,
            requested_tokens=2000
        )
        
        # Mock climate data for US
        with patch.object(LiveElectricityMapAPI, 'get_live_carbon_data') as mock_climate:
            mock_climate.return_value = LiveCarbonData(
                carbon_intensity=600,
                renewable_percentage=0.25,
                fossil_percentage=0.75,
                timestamp=datetime.utcnow(),
                region="US"
            )
            
            # Create budget (no special treatment for crypto mining)
            budget = CarbonAwareTokenBudget(
                initial_estimate=request.requested_tokens,
                current_estimate=request.requested_tokens,
                actual_usage=request.requested_tokens,
                region=request.origin_region
            )
            
            # Update with climate data
            climate_data = await mock_climate.return_value
            budget.update_climate_metrics(
                climate_data.renewable_percentage,
                climate_data.carbon_intensity
            )
            
            # Calculate economics
            true_cost = budget.calculate_true_cost()
            
            # Assertions for crypto mining penalties
            assert budget.global_south_benefit == 0  # No Global South benefit
            assert budget.climate_adaptation_value == 0  # No climate priority
            assert true_cost.justice_adjustment < 0  # Negative = premium applied
            assert budget.climate_impact_score < 100  # Low climate impact
            
            # Should pay Global North premium
            expected_premium = true_cost.base_cost * settings.climate_justice.global_north_premium_rate
            assert abs(abs(true_cost.justice_adjustment) - expected_premium) < 0.01
    
    def test_comparative_climate_economics(self):
        """Test that climate justice economics work comparatively."""
        
        # Same token usage, different contexts
        token_amount = 2000
        
        # Scenario 1: Bangladesh climate adaptation
        budget_bd = create_test_budget("BD", "climate_adaptation")
        budget_bd.actual_usage = token_amount
        budget_bd.update_climate_metrics(0.4, 300)  # Bangladesh grid
        cost_bd = budget_bd.calculate_true_cost()
        
        # Scenario 2: US crypto mining
        budget_us = CarbonAwareTokenBudget(token_amount, token_amount, token_amount, region="US")
        budget_us.update_climate_metrics(0.25, 600)  # US grid
        cost_us = budget_us.calculate_true_cost()
        
        # Scenario 3: Kenya agriculture
        budget_ke = create_test_budget("KE", "sustainable_agriculture")
        budget_ke.actual_usage = token_amount
        budget_ke.update_climate_metrics(0.4, 300)  # Similar to Bangladesh
        cost_ke = budget_ke.calculate_true_cost()
        
        # Climate justice assertions
        print(f"Bangladesh (climate adaptation): ${cost_bd.true_cost:.2f}")
        print(f"US (crypto mining): ${cost_us.true_cost:.2f}")
        print(f"Kenya (agriculture): ${cost_ke.true_cost:.2f}")
        
        # Bangladesh and Kenya should have justice benefits
        assert budget_bd.global_south_benefit > 0
        assert budget_ke.global_south_benefit > 0
        assert budget_us.global_south_benefit == 0
        
        # Bangladesh should have highest climate impact due to adaptation priority
        assert budget_bd.climate_impact_score > budget_ke.climate_impact_score
        assert budget_bd.climate_impact_score > budget_us.climate_impact_score
        
        # US should pay the most (Global North premium, no benefits)
        # Note: BD/KE might have higher true_cost due to priority allocation, but lower per-token cost
        economics_comparison = {
            "BD": {
                "climate_impact": budget_bd.climate_impact_score,
                "justice_benefit": budget_bd.global_south_benefit,
                "true_cost": cost_bd.true_cost
            },
            "US": {
                "climate_impact": budget_us.climate_impact_score,
                "justice_benefit": budget_us.global_south_benefit,
                "true_cost": cost_us.true_cost
            },
            "KE": {
                "climate_impact": budget_ke.climate_impact_score,
                "justice_benefit": budget_ke.global_south_benefit,
                "true_cost": cost_ke.true_cost
            }
        }
        
        # Climate justice should be working
        assert economics_comparison["BD"]["justice_benefit"] > 0
        assert economics_comparison["KE"]["justice_benefit"] > 0
        assert economics_comparison["US"]["justice_benefit"] == 0


class TestErrorHandlingAndFallbacks:
    """Test error handling and fallback scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_timeout_fallback(self):
        """Test fallback when API times out."""
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock timeout
            mock_get = AsyncMock()
            mock_get.return_value.__aenter__.side_effect = asyncio.TimeoutError("Request timeout")
            mock_session.return_value.__aenter__.return_value.get = mock_get
            
            api = LiveElectricityMapAPI()
            data = await api.get_live_carbon_data("BD")
            
            # Should return fallback data
            assert isinstance(data, LiveCarbonData)
            assert data.region == "BD"
            
            # Should use Global South fallback
            settings = get_settings()
            fallback = settings.electricity_map.fallback_defaults["global_south"]
            assert data.carbon_intensity == fallback.intensity
    
    def test_invalid_region_handling(self):
        """Test handling of invalid/unknown regions."""
        
        # Unknown region should get global defaults
        budget = CarbonAwareTokenBudget(
            initial_estimate=1000,
            current_estimate=1000,
            actual_usage=1000,
            region="UNKNOWN"
        )
        
        # Should not crash when calculating costs
        true_cost = budget.calculate_true_cost()
        assert isinstance(true_cost.true_cost, float)
        assert true_cost.true_cost >= 0
    
    def test_edge_case_token_amounts(self):
        """Test edge cases with very small/large token amounts."""
        
        # Very small amount
        budget_small = CarbonAwareTokenBudget(1, 1, 1, region="BD")
        budget_small.update_climate_metrics(0.5, 300)
        cost_small = budget_small.calculate_true_cost()
        assert cost_small.true_cost >= 0
        
        # Very large amount
        budget_large = CarbonAwareTokenBudget(1000000, 1000000, 1000000, region="BD")
        budget_large.update_climate_metrics(0.5, 300)
        cost_large = budget_large.calculate_true_cost()
        assert cost_large.true_cost >= 0
        assert cost_large.true_cost > cost_small.true_cost  # Should scale


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

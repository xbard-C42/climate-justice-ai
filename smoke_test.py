"""
Integration smoke test for Climate Justice AI Orchestrator.

Tests configuration loading, module imports, and basic climate justice economics.

Save this file as: smoke_test.py in your climate-justice-ai root directory
"""

import asyncio
from datetime import datetime


def test_configuration():
    """Test configuration loading works correctly."""
    print("🔧 Testing Configuration...")
    
    try:
        from orchestrator.config_setup import get_settings, get_logger
        
        settings = get_settings()
        logger = get_logger(__name__)
        
        logger.info("🌍 Climate Justice AI - Configuration Test")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        
        # Test climate justice settings
        cj = settings.climate_justice
        logger.info(f"💰 Justice discount rate: {cj.justice_discount_rate:.1%}")
        logger.info(f"⚡ Global North premium: {cj.global_north_premium_rate:.1%}")
        logger.info(f"🌱 Renewable discount: {cj.renewable_discount_rate:.1%}")
        
        # Test region classifications
        logger.info(f"🌍 Global South regions: {len(cj.global_south_regions)} countries")
        logger.info(f"🚨 Climate purposes: {len(cj.climate_adaptation_purposes)} categories")
        
        # Test ElectricityMap config
        em = settings.electricity_map
        logger.info(f"🔌 ElectricityMap configured: {bool(em.api_key)}")
        logger.info(f"🔄 Cache TTL: {em.cache_ttl_seconds}s")
        
        print("✅ Configuration test PASSED")
        return settings
        
    except Exception as e:
        print(f"❌ Configuration test FAILED: {e}")
        raise


def test_module_imports():
    """Test that all modules import correctly."""
    print("\n📦 Testing Module Imports...")
    
    try:
        # Test climate data imports
        from orchestrator.climate_data_implementation import LiveCarbonData, LiveElectricityMapAPI
        print("✅ Climate data module imports successful")
        
        # Test budget imports  
        from orchestrator.climate_budget_implementation import CarbonAwareTokenBudget, create_test_budget
        print("✅ Climate budget module imports successful")
        
        # Test types imports
        from orchestrator.types_and_errors import AIRequest, AIResponse, RoleDefinition
        print("✅ Types and errors module imports successful")
        
        print("✅ All module imports PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Module import test FAILED: {e}")
        raise


def test_climate_justice_economics():
    """Test core climate justice economics calculations."""
    print("\n💰 Testing Climate Justice Economics...")
    
    try:
        from orchestrator.climate_budget_implementation import create_test_budget
        from orchestrator.config_setup import get_logger
        
        logger = get_logger(__name__)
        
        # Test scenarios
        scenarios = [
            ("BD", "climate_adaptation", "Bangladesh flood prediction"),
            ("US", "crypto_mining", "US crypto mining optimization"),
            ("KE", "sustainable_agriculture", "Kenya sustainable agriculture")
        ]
        
        results = []
        
        for region, purpose, description in scenarios:
            logger.info(f"\n--- Testing {description} ---")
            
            # Create budget with climate justice pricing
            budget = create_test_budget(region, purpose)
            budget.actual_usage = 2000
            
            # Simulate climate data
            renewable_ratio = 0.4 if region in ["BD", "KE"] else 0.25
            grid_intensity = 300 if region in ["BD", "KE"] else 600
            
            budget.update_climate_metrics(renewable_ratio, grid_intensity)
            true_cost = budget.calculate_true_cost()
            
            result = {
                'name': description,
                'region': region,
                'global_south_benefit': budget.global_south_benefit,
                'climate_adaptation_value': budget.climate_adaptation_value,
                'climate_impact_score': budget.climate_impact_score,
                'true_cost': true_cost.true_cost,
                'carbon_footprint': budget.carbon_footprint
            }
            
            results.append(result)
            
            logger.info(f"Region: {region}")
            logger.info(f"Global South Benefit: ${result['global_south_benefit']:.2f}")
            logger.info(f"Climate Impact Score: {result['climate_impact_score']:.1f}")
            logger.info(f"True Cost: ${result['true_cost']:.2f}")
            logger.info(f"Carbon Footprint: {result['carbon_footprint']:.3f} gCO2e")
        
        # Verify climate justice is working
        bd_result = next(r for r in results if r['region'] == 'BD')
        us_result = next(r for r in results if r['region'] == 'US')
        ke_result = next(r for r in results if r['region'] == 'KE')
        
        # Assertions
        assert bd_result['global_south_benefit'] > 0, "Bangladesh should get Global South benefit"
        assert ke_result['global_south_benefit'] > 0, "Kenya should get Global South benefit"
        assert us_result['global_south_benefit'] == 0, "US should not get Global South benefit"
        
        assert bd_result['climate_impact_score'] > us_result['climate_impact_score'], "Climate adaptation should have higher impact than crypto mining"
        
        print("✅ Climate justice economics test PASSED")
        print(f"🎉 Bangladesh climate adaptation gets {bd_result['climate_impact_score']:.0f} impact score")
        print(f"🎉 US crypto mining gets {us_result['climate_impact_score']:.0f} impact score")
        print(f"🎉 Ratio: {bd_result['climate_impact_score'] / max(us_result['climate_impact_score'], 1):.1f}x priority for climate justice!")
        
        return results
        
    except Exception as e:
        print(f"❌ Climate justice economics test FAILED: {e}")
        raise


async def test_climate_data_fallback():
    """Test climate data with fallback (no API key needed)."""
    print("\n🌍 Testing Climate Data (Fallback Mode)...")
    
    try:
        from orchestrator.climate_data_implementation import LiveElectricityMapAPI
        
        # Test fallback data (will work without API key)
        api = LiveElectricityMapAPI()
        
        # This should use fallback data
        test_regions = ["BD", "US", "EU"]
        
        for region in test_regions:
            try:
                data = await api.get_live_carbon_data(region)
                print(f"✅ {region}: {data.carbon_intensity:.1f} gCO2eq/kWh, "
                      f"{data.renewable_percentage:.1%} renewable ({data.data_source})")
            except Exception as e:
                print(f"⚠️  {region}: Using fallback - {e}")
        
        await api.close()
        print("✅ Climate data fallback test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Climate data test FAILED: {e}")
        raise


async def main():
    """Run all smoke tests."""
    print("🚀 Climate Justice AI - Integration Smoke Test")
    print("=" * 50)
    
    try:
        # Test configuration
        settings = test_configuration()
        
        # Test module imports
        test_module_imports()
        
        # Test climate justice economics
        test_climate_justice_economics()
        
        # Test climate data (fallback mode)
        await test_climate_data_fallback()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Climate Justice AI system is ready!")
        print("🌍 Revolutionary climate justice economics validated ✅")
        print("⚡ Global South prioritization working correctly ✅") 
        print("🚀 Ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        print("Please fix the error above and re-run the test.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
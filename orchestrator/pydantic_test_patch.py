# Quick patch to update test configuration usage for Pydantic settings

# In tests/test_climate_integration.py, update these lines:

# OLD:
# from orchestrator.config import get_settings, get_logger
# settings = get_settings()

# NEW:
from orchestrator.config import settings, get_logger

# OLD:
# if region in settings.climate_justice.global_south_regions:
#     fallback = settings.electricity_map.fallback_defaults["global_south"]
# else:
#     fallback = settings.electricity_map.fallback_defaults["global_north"]

# NEW:
if region in settings.climate_justice.global_south_regions:
    fallback = settings.electricity_map.fallback_defaults_global_south
else:
    fallback = settings.electricity_map.fallback_defaults_global_north

# And access properties as:
# fallback.intensity (instead of fallback["carbon_intensity"])
# fallback.renewable_ratio (instead of fallback["renewable_percentage"])

# Example complete update for one test:
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
        
        # Updated: Use Pydantic settings structure
        fallback = settings.electricity_map.fallback_defaults_global_south
        assert data.carbon_intensity == fallback.intensity
        assert data.renewable_percentage == fallback.renewable_ratio

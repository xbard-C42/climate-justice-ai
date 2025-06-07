"""
Live climate data integration for Climate Justice AI Orchestrator.

Integrates with ElectricityMap API to get real-time carbon intensity and renewable energy data.
Implements caching, fallbacks, and rate limiting for production reliability.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import json
import logging

import aiohttp
from pydantic import BaseModel, Field

from .config_setup import get_settings, get_logger

logger = get_logger(__name__)


class LiveCarbonData(BaseModel):
    """Live carbon and renewable energy data for a region."""
    
    carbon_intensity: float = Field(..., description="Carbon intensity in gCO2eq/kWh")
    renewable_percentage: float = Field(..., ge=0.0, le=1.0, description="Renewable energy ratio (0-1)")
    fossil_percentage: float = Field(..., ge=0.0, le=1.0, description="Fossil fuel ratio (0-1)")
    timestamp: datetime = Field(..., description="Data timestamp")
    region: str = Field(..., description="Region/country code")
    data_source: str = Field(default="electricitymap", description="Data source")
    
    # Additional metadata
    nuclear_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nuclear energy ratio")
    hydro_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Hydro energy ratio")
    wind_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Wind energy ratio")
    solar_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Solar energy ratio")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ElectricityMapAPIError(Exception):
    """Base exception for ElectricityMap API errors."""
    pass


class ElectricityMapRateLimitError(ElectricityMapAPIError):
    """Raised when API rate limit is exceeded."""
    pass


class ElectricityMapDataUnavailableError(ElectricityMapAPIError):
    """Raised when data is not available for a region."""
    pass


class LiveElectricityMapAPI:
    """
    Integration with ElectricityMap API for live carbon intensity and renewable energy data.
    
    Features:
    - Automatic caching with TTL
    - Rate limiting protection
    - Fallback to default values
    - Error handling and retries
    - Background cache warming
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.config = self.settings.electricity_map
        
        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request_time = 0.0
        self._request_count = 0
        self._request_window_start = time.time()
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Initialized ElectricityMap API client (cache_ttl={self.config.cache_ttl_seconds}s)")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _is_cached(self, region: str) -> bool:
        """Check if data for region is cached and still valid."""
        if region not in self._cache:
            return False
        
        cache_entry = self._cache[region]
        cache_time = cache_entry.get('timestamp', 0)
        return time.time() - cache_time < self.config.cache_ttl_seconds
    
    def _get_cached(self, region: str) -> Optional[LiveCarbonData]:
        """Get cached data for region if available and valid."""
        if not self._is_cached(region):
            return None
        
        try:
            cache_entry = self._cache[region]
            return LiveCarbonData(**cache_entry['data'])
        except Exception as e:
            logger.warning(f"Invalid cache entry for region {region}: {e}")
            # Remove invalid cache entry
            if region in self._cache:
                del self._cache[region]
            return None
    
    def _cache_data(self, region: str, data: LiveCarbonData):
        """Cache data for region."""
        self._cache[region] = {
            'timestamp': time.time(),
            'data': data.dict()
        }
        logger.debug(f"Cached carbon data for region {region}")
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset rate limit window if needed
        if current_time - self._request_window_start > 60:  # 1 minute window
            self._request_count = 0
            self._request_window_start = current_time
        
        # Check rate limit
        if self._request_count >= self.config.requests_per_minute:
            wait_time = 60 - (current_time - self._request_window_start)
            logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            # Reset after waiting
            self._request_count = 0
            self._request_window_start = time.time()
        
        self._request_count += 1
        self._last_request_time = current_time
    
    async def get_live_carbon_data(self, region: str) -> LiveCarbonData:
        """
        Get live carbon intensity and renewable energy data for a region.
        
        Args:
            region: ISO country code (e.g., 'US', 'EU', 'BD')
            
        Returns:
            LiveCarbonData with current carbon intensity and renewable ratios
            
        Raises:
            ElectricityMapAPIError: If API request fails after retries
        """
        # Check cache first
        cached_data = self._get_cached(region)
        if cached_data:
            logger.debug(f"Using cached carbon data for region {region}")
            return cached_data
        
        # Ensure session is available
        await self._ensure_session()
        
        # Get fresh data from API
        try:
            data = await self._fetch_live_data(region)
            self._cache_data(region, data)
            logger.info(f"Fetched live carbon data for region {region}: {data.carbon_intensity:.1f} gCO2eq/kWh")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch live data for region {region}: {e}")
            # Return fallback data
            return self._get_fallback_data(region)
    
    async def _fetch_live_data(self, region: str) -> LiveCarbonData:
        """Fetch live data from ElectricityMap API with retries."""
        
        for attempt in range(self.config.max_retries):
            try:
                await self._check_rate_limit()
                
                # Fetch carbon intensity
                carbon_intensity = await self._fetch_carbon_intensity(region)
                
                # Fetch power breakdown for renewable percentage
                power_breakdown = await self._fetch_power_breakdown(region)
                
                # Calculate renewable percentage
                renewable_percentage = self._calculate_renewable_percentage(power_breakdown)
                fossil_percentage = 1.0 - renewable_percentage
                
                # Extract individual source percentages
                nuclear_pct = power_breakdown.get('nuclear', 0) / max(sum(power_breakdown.values()), 1)
                hydro_pct = power_breakdown.get('hydro', 0) / max(sum(power_breakdown.values()), 1)
                wind_pct = power_breakdown.get('wind', 0) / max(sum(power_breakdown.values()), 1)
                solar_pct = power_breakdown.get('solar', 0) / max(sum(power_breakdown.values()), 1)
                
                return LiveCarbonData(
                    carbon_intensity=carbon_intensity,
                    renewable_percentage=renewable_percentage,
                    fossil_percentage=fossil_percentage,
                    timestamp=datetime.utcnow(),
                    region=region,
                    nuclear_percentage=nuclear_pct,
                    hydro_percentage=hydro_pct,
                    wind_percentage=wind_pct,
                    solar_percentage=solar_pct
                )
                
            except ElectricityMapRateLimitError:
                # Don't retry on rate limit, just wait
                raise
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} failed for region {region}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise ElectricityMapAPIError(f"Failed to fetch data for region {region} after {self.config.max_retries} attempts: {e}")
    
    async def _fetch_carbon_intensity(self, region: str) -> float:
        """Fetch carbon intensity for region."""
        url = f"{self.config.base_url}/carbon-intensity/latest"
        params = {"zone": region}
        
        async with self._session.get(url, params=params) as response:
            if response.status == 429:
                raise ElectricityMapRateLimitError("Rate limit exceeded")
            elif response.status == 404:
                raise ElectricityMapDataUnavailableError(f"Data not available for region {region}")
            elif response.status != 200:
                response.raise_for_status()
            
            data = await response.json()
            
            if 'data' not in data or 'carbonIntensity' not in data['data']:
                raise ElectricityMapAPIError(f"Invalid carbon intensity response for region {region}")
            
            return float(data['data']['carbonIntensity'])
    
    async def _fetch_power_breakdown(self, region: str) -> Dict[str, float]:
        """Fetch power generation breakdown for region."""
        url = f"{self.config.base_url}/power-breakdown/latest"
        params = {"zone": region}
        
        async with self._session.get(url, params=params) as response:
            if response.status == 429:
                raise ElectricityMapRateLimitError("Rate limit exceeded")
            elif response.status == 404:
                # Some regions might not have breakdown data, return empty
                logger.warning(f"Power breakdown not available for region {region}")
                return {}
            elif response.status != 200:
                response.raise_for_status()
            
            data = await response.json()
            
            if 'data' not in data or 'powerConsumptionBreakdown' not in data['data']:
                logger.warning(f"No power breakdown data for region {region}")
                return {}
            
            breakdown = data['data']['powerConsumptionBreakdown']
            # Convert None values to 0
            return {k: float(v) if v is not None else 0.0 for k, v in breakdown.items()}
    
    def _calculate_renewable_percentage(self, power_breakdown: Dict[str, float]) -> float:
        """Calculate renewable energy percentage from power breakdown."""
        renewable_sources = {
            'wind', 'solar', 'hydro', 'geothermal', 'biomass'
            # Note: Nuclear is often considered low-carbon but not renewable
        }
        
        total_renewable = sum(
            power_breakdown.get(source, 0) 
            for source in renewable_sources
        )
        
        total_consumption = sum(power_breakdown.values())
        
        if total_consumption <= 0:
            return 0.0
        
        return min(total_renewable / total_consumption, 1.0)  # Cap at 100%
    
    def _get_fallback_data(self, region: str) -> LiveCarbonData:
        """Get fallback data when API is unavailable."""
        fallback = self.config.fallback_regions.get(
            region, 
            {"carbon_intensity": 500, "renewable_percentage": 0.2}  # Global average
        )
        
        logger.warning(f"Using fallback carbon data for region {region}: {fallback}")
        
        return LiveCarbonData(
            carbon_intensity=fallback["carbon_intensity"],
            renewable_percentage=fallback["renewable_percentage"],
            fossil_percentage=1.0 - fallback["renewable_percentage"],
            timestamp=datetime.utcnow(),
            region=region,
            data_source="fallback"
        )
    
    async def warm_cache(self, regions: list[str]):
        """Pre-load cache for multiple regions (background task)."""
        logger.info(f"Warming cache for {len(regions)} regions")
        
        async def fetch_region(region: str):
            try:
                await self.get_live_carbon_data(region)
            except Exception as e:
                logger.warning(f"Failed to warm cache for region {region}: {e}")
        
        # Fetch regions concurrently with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def fetch_with_semaphore(region: str):
            async with semaphore:
                await fetch_region(region)
        
        await asyncio.gather(*[fetch_with_semaphore(region) for region in regions])
        logger.info("Cache warming completed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        current_time = time.time()
        valid_entries = sum(
            1 for region, entry in self._cache.items()
            if current_time - entry.get('timestamp', 0) < self.config.cache_ttl_seconds
        )
        
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / max(len(self._cache), 1),
            "request_count": self._request_count,
            "last_request_time": self._last_request_time
        }


# Module-level API instance for easy access
_api_instance: Optional[LiveElectricityMapAPI] = None

async def get_climate_api() -> LiveElectricityMapAPI:
    """Get shared ElectricityMap API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = LiveElectricityMapAPI()
        await _api_instance._ensure_session()
    return _api_instance

async def close_climate_api():
    """Close shared API instance."""
    global _api_instance
    if _api_instance:
        await _api_instance.close()
        _api_instance = None


# Development/testing utilities
async def test_region_data(region: str) -> LiveCarbonData:
    """Test function to fetch data for a specific region."""
    async with LiveElectricityMapAPI() as api:
        return await api.get_live_carbon_data(region)

if __name__ == "__main__":
    # Test script
    async def main():
        """Test the ElectricityMap API integration."""
        test_regions = ["US", "EU", "BD", "KE", "CN"]
        
        async with LiveElectricityMapAPI() as api:
            for region in test_regions:
                try:
                    data = await api.get_live_carbon_data(region)
                    print(f"{region}: {data.carbon_intensity:.1f} gCO2eq/kWh, "
                          f"{data.renewable_percentage:.1%} renewable")
                except Exception as e:
                    print(f"{region}: Error - {e}")
            
            # Print cache stats
            stats = api.get_cache_stats()
            print(f"\nCache stats: {stats}")
    
    asyncio.run(main())

"""
Climate Justice Token Budget Management.

Implements carbon-aware token budgets with climate justice economics:
- Global South prioritization and discounts
- Climate adaptation value tracking  
- Carbon footprint calculation
- Renewable energy incentives
- Value creation beyond energy costs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from pydantic import BaseModel

from .config_setup import get_settings, get_logger
from .climate_data_implementation import LiveCarbonData

logger = get_logger(__name__)


@dataclass
class ClimateAdjustedCost:
    """Complete cost breakdown including climate and justice factors."""
    base_cost: float
    carbon_cost: float
    justice_adjustment: float  # Positive = discount, Negative = premium
    renewable_discount: float
    true_cost: float
    
    @property
    def climate_premium_ratio(self) -> float:
        """How much more/less expensive due to climate factors."""
        if self.base_cost == 0:
            return 0.0
        return (self.true_cost - self.base_cost) / self.base_cost


@dataclass
class ClimateImpact:
    """Climate impact assessment for token operations."""
    carbon_footprint: float  # gCO2e
    climate_benefit: float   # Climate benefit score
    net_climate_impact: float  # Positive = net benefit
    
    @property
    def is_climate_positive(self) -> bool:
        return self.net_climate_impact > 0


@dataclass
class ValueCreationEvent:
    """Track events that create economic value beyond energy costs."""
    timestamp: datetime
    event_type: str  # 'quality_bonus', 'collaboration_bonus', 'efficiency_gain', 'justice_impact'
    tokens_involved: int
    value_created: float
    quality_metrics: Dict[str, float]
    description: str


@dataclass
class CarbonAwareTokenBudget:
    """
    Enhanced token budget with climate justice integration.
    
    Tracks both traditional token economics and climate justice metrics:
    - Carbon footprint and renewable energy usage
    - Global South economic benefits
    - Climate adaptation value creation
    - Justice-adjusted pricing
    """
    
    # Base token economics
    initial_estimate: int
    current_estimate: int
    actual_usage: int = 0
    quality_multiplier: float = 1.0
    collaboration_bonus: float = 0.0
    
    # Climate metrics
    carbon_footprint: float = 0.0  # gCO2e per session
    renewable_energy_ratio: float = 0.0  # % renewable energy used
    grid_carbon_intensity: float = 500.0  # gCO2e/kWh (global average)
    climate_impact_score: float = 0.0
    
    # Justice metrics
    global_south_benefit: float = 0.0  # Economic value created for Global South
    community_impact: float = 0.0  # Local community benefit
    climate_adaptation_value: float = 0.0  # Value for adaptation efforts
    region: str = "global"
    
    # Value tracking
    market_value: float = 0.0  # Market-determined value
    value_creation_events: List[ValueCreationEvent] = field(default_factory=list)
    efficiency_history: List[float] = field(default_factory=list)
    
    # Settings
    _settings: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize settings if not provided."""
        if self._settings is None:
            self._settings = get_settings()
    
    def apply_global_south_discount(self, discount_rate: float = 0.3):
        """Apply Global South justice discount."""
        if self.region in self._settings.climate_justice.global_south_regions:
            self.global_south_benefit = self.initial_estimate * discount_rate
            logger.debug(f"Applied Global South discount: ${self.global_south_benefit:.2f}")
    
    def apply_priority_multiplier(self, multiplier: float):
        """Apply climate adaptation priority multiplier."""
        original_estimate = self.current_estimate
        self.current_estimate = int(self.current_estimate * multiplier)
        self.climate_adaptation_value = self.current_estimate - original_estimate
        logger.debug(f"Applied priority multiplier {multiplier}x: +{self.climate_adaptation_value} tokens")
    
    def update_climate_metrics(self, 
                             renewable_ratio: float,
                             grid_intensity: float,
                             execution_time_seconds: float = 1.0):
        """Update climate metrics based on actual usage."""
        self.renewable_energy_ratio = renewable_ratio
        self.grid_carbon_intensity = grid_intensity
        
        # Calculate carbon footprint
        # Simplified model: tokens * time * grid_intensity * energy_factor
        energy_per_token_kwh = 0.0001  # Rough estimate: 0.1 Wh per token
        energy_usage_kwh = self.actual_usage * energy_per_token_kwh * execution_time_seconds
        
        # Carbon footprint from non-renewable energy
        self.carbon_footprint = energy_usage_kwh * grid_intensity * (1.0 - renewable_ratio)
        
        # Calculate climate impact score
        self._calculate_climate_impact_score()
        
        logger.debug(f"Updated climate metrics: {self.carbon_footprint:.3f} gCO2e, "
                    f"{renewable_ratio:.1%} renewable")
    
    def _calculate_climate_impact_score(self):
        """Calculate holistic climate impact score."""
        # Negative impact from carbon emissions
        carbon_penalty = self.carbon_footprint * -10  # -10 points per gram CO2e
        
        # Positive impact from renewable energy usage
        renewable_bonus = self.renewable_energy_ratio * self.actual_usage * 0.1
        
        # Justice impact bonus
        justice_bonus = (self.global_south_benefit + self.climate_adaptation_value) * 10
        
        # Efficiency bonus
        if self.actual_usage > 0 and self.initial_estimate > 0:
            efficiency_ratio = self.initial_estimate / self.actual_usage
            efficiency_bonus = max(0, (efficiency_ratio - 1.0) * 50)  # Bonus for under-budget
        else:
            efficiency_bonus = 0
        
        self.climate_impact_score = (
            carbon_penalty + renewable_bonus + justice_bonus + efficiency_bonus
        )
    
    def calculate_true_cost(self) -> ClimateAdjustedCost:
        """Calculate cost including climate and justice factors."""
        settings = self._settings.climate_justice
        
        # Base energy cost
        base_cost = self.actual_usage * settings.energy_cost_per_token
        
        # Carbon cost (pricing externalities)
        carbon_cost = self.carbon_footprint * settings.carbon_price_per_gram
        
        # Justice adjustment
        if self.global_south_benefit > 0:
            # Discount for Global South benefit
            justice_adjustment = self.global_south_benefit * settings.justice_discount_rate
        else:
            # Premium for Global North consumption
            justice_adjustment = -(base_cost * settings.global_north_premium_rate)
        
        # Renewable energy discount
        renewable_discount = (
            self.renewable_energy_ratio * settings.renewable_discount_rate * base_cost
        )
        
        true_cost = max(0.0, base_cost + carbon_cost + justice_adjustment - renewable_discount)
        
        return ClimateAdjustedCost(
            base_cost=base_cost,
            carbon_cost=carbon_cost,
            justice_adjustment=justice_adjustment,
            renewable_discount=renewable_discount,
            true_cost=true_cost
        )
    
    def update_actual_usage(self, tokens_used: int, carbon_footprint: float):
        """Update actual usage and recalculate metrics."""
        self.actual_usage = tokens_used
        self.carbon_footprint = carbon_footprint
        
        # Track efficiency
        if self.initial_estimate > 0:
            efficiency = tokens_used / self.initial_estimate
            self.efficiency_history.append(efficiency)
            
            # Keep only recent efficiency data
            if len(self.efficiency_history) > 10:
                self.efficiency_history = self.efficiency_history[-10:]
        
        # Recalculate climate impact
        self._calculate_climate_impact_score()
        
        # Update market value
        self._update_market_value()
    
    def _update_market_value(self):
        """Update market value based on quality and collaboration."""
        # Current: simple multiplier, Future: full market calculation
        base_value = self.actual_usage
        quality_bonus = base_value * (self.quality_multiplier - 1.0)
        collaboration_bonus_value = base_value * self.collaboration_bonus
        
        # Justice value multiplier
        justice_multiplier = 1.0
        if self.global_south_benefit > 0:
            justice_multiplier += 0.5  # 50% bonus for Global South benefit
        if self.climate_adaptation_value > 0:
            justice_multiplier += 1.0  # 100% bonus for climate adaptation
        
        self.market_value = (base_value + quality_bonus + collaboration_bonus_value) * justice_multiplier
    
    def add_value_creation_event(self, 
                               event_type: str,
                               value_created: float,
                               quality_metrics: Dict[str, float] = None,
                               description: str = ""):
        """Record a value creation event."""
        event = ValueCreationEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            tokens_involved=self.actual_usage,
            value_created=value_created,
            quality_metrics=quality_metrics or {},
            description=description
        )
        
        self.value_creation_events.append(event)
        logger.debug(f"Added value creation event: {event_type} (+${value_created:.2f})")
    
    def get_climate_efficiency(self) -> float:
        """Calculate climate efficiency: value per unit carbon."""
        if self.carbon_footprint <= 0:
            return float('inf')  # Perfect efficiency (no carbon)
        
        total_value = (
            self.market_value +
            self.global_south_benefit * 10 +  # Justice value multiplied
            self.climate_adaptation_value * 5   # Adaptation value
        )
        
        return total_value / self.carbon_footprint
    
    def get_roi(self) -> float:
        """Return on Investment - foundational for token economics."""
        if self.actual_usage == 0:
            return 0.0
        
        true_cost = self.calculate_true_cost().true_cost
        if true_cost <= 0:
            return float('inf')
        
        return (self.market_value - true_cost) / true_cost
    
    def get_efficiency_trend(self) -> str:
        """Get recent efficiency trend."""
        if len(self.efficiency_history) < 2:
            return "insufficient_data"
        
        recent_avg = sum(self.efficiency_history[-3:]) / len(self.efficiency_history[-3:])
        older_avg = sum(self.efficiency_history[:-3]) / max(len(self.efficiency_history[:-3]), 1)
        
        if recent_avg < older_avg * 0.95:
            return "improving"  # Using fewer tokens than estimated
        elif recent_avg > older_avg * 1.05:
            return "declining"  # Using more tokens than estimated
        else:
            return "stable"
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary for logging/monitoring."""
        true_cost = self.calculate_true_cost()
        
        return {
            "session_id": getattr(self, 'session_id', 'unknown'),
            "region": self.region,
            "tokens": {
                "estimated": self.initial_estimate,
                "actual": self.actual_usage,
                "efficiency": self.actual_usage / max(self.initial_estimate, 1)
            },
            "costs": {
                "base_cost": true_cost.base_cost,
                "true_cost": true_cost.true_cost,
                "climate_premium": true_cost.climate_premium_ratio
            },
            "climate": {
                "carbon_footprint": self.carbon_footprint,
                "renewable_ratio": self.renewable_energy_ratio,
                "climate_impact_score": self.climate_impact_score,
                "climate_efficiency": self.get_climate_efficiency()
            },
            "justice": {
                "global_south_benefit": self.global_south_benefit,
                "climate_adaptation_value": self.climate_adaptation_value,
                "roi": self.get_roi()
            },
            "trends": {
                "efficiency_trend": self.get_efficiency_trend(),
                "value_events": len(self.value_creation_events)
            }
        }


class ClimateJusticeMetrics:
    """Aggregate climate justice metrics across sessions."""
    
    def __init__(self):
        self.daily_metrics = {}
        self.session_history = []
    
    def record_session(self, budget: CarbonAwareTokenBudget):
        """Record session metrics for aggregation."""
        today = datetime.utcnow().date()
        
        if today not in self.daily_metrics:
            self.daily_metrics[today] = {
                'total_sessions': 0,
                'total_tokens': 0,
                'total_carbon_footprint': 0.0,
                'total_global_south_benefit': 0.0,
                'total_climate_adaptation_value': 0.0,
                'adaptation_sessions': 0,
                'global_south_sessions': 0
            }
        
        daily = self.daily_metrics[today]
        daily['total_sessions'] += 1
        daily['total_tokens'] += budget.actual_usage
        daily['total_carbon_footprint'] += budget.carbon_footprint
        daily['total_global_south_benefit'] += budget.global_south_benefit
        daily['total_climate_adaptation_value'] += budget.climate_adaptation_value
        
        if budget.climate_adaptation_value > 0:
            daily['adaptation_sessions'] += 1
        if budget.global_south_benefit > 0:
            daily['global_south_sessions'] += 1
        
        # Keep session summary for analysis
        self.session_history.append(budget.to_summary_dict())
        
        # Limit history size
        if len(self.session_history) > 1000:
            self.session_history = self.session_history[-500:]  # Keep last 500
    
    def get_daily_summary(self, date=None) -> Dict[str, Any]:
        """Get daily metrics summary."""
        target_date = date or datetime.utcnow().date()
        
        if target_date not in self.daily_metrics:
            return {
                'date': target_date.isoformat(),
                'no_data': True
            }
        
        daily = self.daily_metrics[target_date]
        
        return {
            'date': target_date.isoformat(),
            'sessions': daily['total_sessions'],
            'total_tokens': daily['total_tokens'],
            'carbon_footprint_g': daily['total_carbon_footprint'],
            'global_south_benefit_usd': daily['total_global_south_benefit'],
            'climate_adaptation_value': daily['total_climate_adaptation_value'],
            'justice_sessions_pct': (
                (daily['adaptation_sessions'] + daily['global_south_sessions']) / 
                max(daily['total_sessions'], 1) * 100
            ),
            'avg_carbon_per_session': daily['total_carbon_footprint'] / max(daily['total_sessions'], 1),
            'climate_positive': daily['total_climate_adaptation_value'] > daily['total_carbon_footprint']
        }


# Module-level metrics tracker
_metrics_tracker: Optional[ClimateJusticeMetrics] = None

def get_metrics_tracker() -> ClimateJusticeMetrics:
    """Get shared metrics tracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = ClimateJusticeMetrics()
    return _metrics_tracker


# Development utilities
def create_test_budget(region: str = "BD", purpose: str = "climate_adaptation") -> CarbonAwareTokenBudget:
    """Create a test budget for development/testing."""
    budget = CarbonAwareTokenBudget(
        initial_estimate=1000,
        current_estimate=1000,
        region=region
    )
    
    # Apply justice pricing based on region and purpose
    if region in get_settings().climate_justice.global_south_regions:
        budget.apply_global_south_discount()
    
    if purpose in get_settings().climate_justice.climate_adaptation_purposes:
        budget.apply_priority_multiplier(3.0)
    
    return budget


if __name__ == "__main__":
    # Test the climate budget system
    def test_climate_budget():
        """Test climate budget calculations."""
        
        print("🌍 Climate Justice Token Budget Test")
        
        # Test scenarios
        scenarios = [
            ("BD", "climate_adaptation", 2000),  # Bangladesh climate adaptation
            ("US", "crypto_mining", 2000),       # US crypto mining
            ("KE", "sustainable_agriculture", 1500)  # Kenya agriculture
        ]
        
        for region, purpose, tokens in scenarios:
            print(f"\n--- {purpose.replace('_', ' ').title()} in {region} ---")
            
            budget = create_test_budget(region, purpose)
            budget.actual_usage = tokens
            
            # Simulate climate data
            renewable_ratio = 0.4 if region in ["BD", "KE"] else 0.25
            grid_intensity = 300 if region in ["BD", "KE"] else 600
            
            budget.update_climate_metrics(renewable_ratio, grid_intensity)
            budget.update_actual_usage(tokens, budget.carbon_footprint)
            
            # Calculate costs
            true_cost = budget.calculate_true_cost()
            
            print(f"Tokens: {tokens}")
            print(f"True Cost: ${true_cost.true_cost:.2f}")
            print(f"Climate Impact Score: {budget.climate_impact_score:.1f}")
            print(f"Global South Benefit: ${budget.global_south_benefit:.2f}")
            print(f"Carbon Footprint: {budget.carbon_footprint:.3f} gCO2e")
            print(f"Climate Efficiency: {budget.get_climate_efficiency():.1f}")
    
    test_climate_budget()

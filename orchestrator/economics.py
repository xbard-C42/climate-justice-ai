# earth_justice/economics.py
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

@dataclass
class EconomicContext:
    economic_tier: Literal["basic_needs", "stable_living", "comfortable", "surplus_wealth"]
    purpose_category: str
    region: str
    cost_of_living_index: float = 1.0
    monthly_usage_tokens: int = 0
    business_usage_ratio: float = 0.0
    student_verified: bool = False
    nonprofit_verified: bool = False
    enterprise_account: bool = False
    annual_income_eur: Optional[int] = None

class UtilityBasedPricingEngine:
    """
    Calculate justice-aware pricing based on utility plateau mathematics and climate justice rules.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config["utility_thresholds"]
        self.tier_multipliers = config["tier_multipliers"]
        self.purpose_adjustments = config["purpose_adjustments"]
        self.cost_of_living = config["cost_of_living"]
        self.verification_discount_rate = config.get("verification_discount", 0.8)

    def calculate_justice_pricing(
        self,
        base_tokens: int,
        ctx: EconomicContext,
        climate_purpose: bool = False
    ) -> Dict[str, float]:
        # 1. Base cost
        base_cost = base_tokens * 0.001

        # 2. Tier multiplier
        economic_multiplier = self.tier_multipliers.get(ctx.economic_tier, 1.0)

        # 3. Purchasing power adjustment
        region_factor = self.cost_of_living.get(ctx.region, ctx.cost_of_living_index)
        if ctx.economic_tier in ["basic_needs", "stable_living"]:
            purchasing_power_adjustment = max(region_factor, 0.3)
        else:
            purchasing_power_adjustment = region_factor

        # 4. Purpose multiplier
        purpose_multiplier = self.purpose_adjustments.get(ctx.purpose_category, 1.0)

        # 5. Usage multiplier via original patterns
        usage_multiplier = self._calculate_usage_multiplier(ctx)

        # 6. Climate multiplier
        climate_multiplier = 0.5 if climate_purpose else 1.0

        # 7. Verification discount
        verification_delta = 0.0
        if ctx.student_verified:
            verification_delta += 0.3
        if ctx.nonprofit_verified:
            verification_delta += 0.4
        if ctx.enterprise_account:
            verification_delta -= 0.5
        # Cap discounts/premiums at ±50%
        verification_multiplier = 1.0 - max(-0.5, min(0.5, verification_delta))

        # 8. Combine multipliers
        total_multiplier = (
            economic_multiplier
            * purchasing_power_adjustment
            * purpose_multiplier
            * usage_multiplier
            * climate_multiplier
            * verification_multiplier
        )

        # 9. Adjusted cost
        adjusted_cost = base_cost * total_multiplier

        # 10. Redistributable surplus
        redistributable_surplus = max(0.0, adjusted_cost - (base_cost * 0.8))

        # 11. Utility impact message
        if ctx.annual_income_eur and ctx.annual_income_eur > self.thresholds.get("comfortable_max", float('inf')):
            utility_impact = f"Zero marginal utility above €{self.thresholds.get('comfortable_max'):,} plateau"
        else:
            utility_impact = f"Applied multiplier {total_multiplier:.2f} based on context."

        return {
            'base_cost': base_cost,
            'adjusted_cost': adjusted_cost,
            'total_multiplier': total_multiplier,
            'redistributable_surplus': redistributable_surplus,
            'utility_impact': utility_impact,
            'breakdown': {
                'economic_multiplier': economic_multiplier,
                'purchasing_power_adjustment': purchasing_power_adjustment,
                'purpose_multiplier': purpose_multiplier,
                'usage_multiplier': usage_multiplier,
                'climate_multiplier': climate_multiplier,
                'verification_multiplier': verification_multiplier
            }
        }

    def _calculate_usage_multiplier(self, ctx: EconomicContext) -> float:
        if ctx.monthly_usage_tokens > 100_000:
            return 1.4
        elif ctx.monthly_usage_tokens > 50_000:
            return 1.2
        elif ctx.monthly_usage_tokens > 10_000:
            return 1.1
        else:
            return 1.0
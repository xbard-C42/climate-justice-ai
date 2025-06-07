# orchestrator/types.py
"""
Type definitions for Climate Justice AI Orchestrator.

Defines core data structures for roles, responses, and AI requests.
Compatible with Pydantic configuration system.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field, validator

from .climate_budget import CarbonAwareTokenBudget


@dataclass
class RoleDefinition:
    """Definition of an AI agent role for the orchestrator."""
    
    role_id: str
    system_prompt: str
    model_name: str = "gpt-4"
    max_response_tokens: int = 2048
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    
    # Climate justice considerations
    priority_level: int = 1  # 1=normal, 2=high, 3=critical
    carbon_budget_multiplier: float = 1.0  # Adjust carbon budget for role
    
    def __post_init__(self):
        """Validate role definition."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")


@dataclass
class RoleResponse:
    """Response from executing an AI role."""
    
    text: str
    tokens: int
    role_id: Optional[str] = None
    execution_time: Optional[float] = None
    
    # Climate justice tracking
    climate_budget: Optional[CarbonAwareTokenBudget] = None
    carbon_footprint: Optional[float] = None
    
    # Quality metrics
    confidence_score: Optional[float] = None
    quality_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.quality_metrics is None:
            self.quality_metrics = {}


class AIRequest(BaseModel):
    """AI request with climate justice context."""
    
    id: str = Field(..., description="Unique request identifier")
    query: str = Field(..., min_length=1, description="The AI query/prompt")
    
    # Geographic and purpose context
    origin_region: str = Field(..., description="ISO country code (e.g., 'BD', 'US')")
    purpose: str = Field(..., description="Purpose category for climate prioritization")
    
    # Request parameters
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Estimated complexity (0-1)")
    requested_tokens: int = Field(default=1000, gt=0, description="Requested token budget")
    priority_level: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    
    # Climate context
    renewable_energy_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Renewable energy availability")
    emergency_level: int = Field(default=0, ge=0, le=4, description="Climate emergency level (0-4)")
    
    # Collaboration context
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous messages")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    deadline: Optional[datetime] = Field(None, description="Response deadline")
    
    @validator('origin_region')
    def validate_region(cls, v):
        """Validate region code format."""
        if len(v) < 2 or len(v) > 3:
            raise ValueError("Region code should be 2-3 character ISO code")
        return v.upper()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AIResponse(BaseModel):
    """AI response with complete climate justice tracking."""
    
    request_id: str = Field(..., description="Original request ID")
    content: str = Field(..., description="AI response content")
    
    # Execution metrics
    tokens_used: int = Field(..., description="Actual tokens consumed")
    execution_time: float = Field(..., description="Execution time in seconds")
    
    # Climate justice metrics
    carbon_footprint: float = Field(..., description="Carbon footprint in gCO2e")
    climate_impact_score: float = Field(..., description="Overall climate impact score")
    renewable_energy_ratio: float = Field(..., description="Renewable energy ratio used")
    
    # Economic metrics
    base_cost: float = Field(..., description="Base energy cost")
    true_cost: float = Field(..., description="Climate-adjusted true cost")
    global_south_benefit: float = Field(default=0.0, description="Economic benefit to Global South")
    climate_adaptation_value: float = Field(default=0.0, description="Climate adaptation value created")
    
    # Quality and collaboration
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response quality score")
    collaboration_bonus: float = Field(default=0.0, description="Collaboration bonus applied")
    
    # Processing details
    processing_mode: Literal["express", "council", "emergency"] = Field(..., description="Processing mode used")
    roles_used: List[str] = Field(default_factory=list, description="AI roles that contributed")
    
    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class TriageDecision:
    """Decision from the triage system about how to process a request."""
    
    mode: Literal["express", "council", "emergency"]
    confidence: float  # 0.0 to 1.0
    recommended_roles: List[str]
    complexity_score: float
    reasoning: str
    
    # Climate justice considerations
    climate_priority: float = 1.0  # Climate urgency multiplier
    justice_priority: float = 1.0  # Global South priority multiplier
    emergency_response: bool = False  # Climate emergency activation


@dataclass
class OrchestrationResult:
    """Result from the orchestrator processing."""
    
    success: bool
    response: Optional[AIResponse]
    error: Optional[str] = None
    
    # Performance metrics
    total_execution_time: Optional[float] = None
    roles_executed: Optional[List[str]] = None
    
    # Climate justice summary
    total_carbon_footprint: Optional[float] = None
    total_climate_impact: Optional[float] = None
    justice_metrics_summary: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.roles_executed is None:
            self.roles_executed = []
        if self.justice_metrics_summary is None:
            self.justice_metrics_summary = {}


# Utility types for climate emergency handling
@dataclass
class ClimateEmergency:
    """Climate emergency event information."""
    
    emergency_id: str
    region: str
    emergency_type: Literal["flood", "drought", "wildfire", "cyclone", "heatwave"]
    severity_level: int  # 1-4 (GDACS scale)
    affected_population: Optional[int] = None
    start_time: Optional[datetime] = None
    description: str = ""
    
    @property
    def priority_multiplier(self) -> float:
        """Calculate priority multiplier based on severity."""
        return {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}.get(self.severity_level, 2.0)


@dataclass
class ResourceAllocation:
    """Resource allocation decision for climate justice."""
    
    allocated_tokens: int
    carbon_budget: float  # gCO2e budget
    renewable_energy_requirement: float  # Minimum renewable energy ratio
    justice_priority: float  # Priority score
    deadline: Optional[datetime] = None
    emergency_override: bool = False


# ---

# orchestrator/errors.py
"""
Custom exceptions for Climate Justice AI Orchestrator.

Defines error hierarchies for different types of failures in the system.
Compatible with Pydantic configuration system.
"""

from typing import Optional, Any


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# LLM Integration Errors
class LLMError(OrchestratorError):
    """Base class for LLM-related errors."""
    pass


class ContextWindowExhaustionError(LLMError):
    """Raised when LLM context window is exhausted."""
    
    def __init__(self, message="Context window exhausted", tokens_used=None, context_limit=None):
        super().__init__(message, {
            'tokens_used': tokens_used,
            'context_limit': context_limit
        })
        self.tokens_used = tokens_used
        self.context_limit = context_limit


class LLMTrafficError(LLMError):
    """Raised when LLM API returns traffic/rate limiting errors."""
    
    def __init__(self, message="LLM API traffic error", retry_after=None):
        super().__init__(message, {'retry_after': retry_after})
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Raised when LLM API request times out."""
    
    def __init__(self, message="LLM API timeout", timeout_seconds=None):
        super().__init__(message, {'timeout_seconds': timeout_seconds})
        self.timeout_seconds = timeout_seconds


class LLMQualityError(LLMError):
    """Raised when LLM response quality is unacceptable."""
    
    def __init__(self, message="LLM response quality too low", quality_score=None):
        super().__init__(message, {'quality_score': quality_score})
        self.quality_score = quality_score


# Climate Data Errors
class ClimateDataError(OrchestratorError):
    """Base class for climate data errors."""
    pass


class ClimateDataUnavailableError(ClimateDataError):
    """Raised when climate data is not available for a region."""
    
    def __init__(self, message="Climate data unavailable", region=None):
        super().__init__(message, {'region': region})
        self.region = region


class ClimateAPIError(ClimateDataError):
    """Raised when climate data API fails."""
    
    def __init__(self, message="Climate API error", api_name=None, status_code=None):
        super().__init__(message, {
            'api_name': api_name,
            'status_code': status_code
        })
        self.api_name = api_name
        self.status_code = status_code


# Context Management Errors
class ContextManagementError(OrchestratorError):
    """Base class for context management errors."""
    pass


class StateTransitionError(ContextManagementError):
    """Raised when state transition fails."""
    
    def __init__(self, message="State transition failed", from_state=None, to_state=None):
        super().__init__(message, {
            'from_state': from_state,
            'to_state': to_state
        })
        self.from_state = from_state
        self.to_state = to_state


class CheckpointError(ContextManagementError):
    """Raised when checkpoint creation/restoration fails."""
    
    def __init__(self, message="Checkpoint operation failed", checkpoint_id=None):
        super().__init__(message, {'checkpoint_id': checkpoint_id})
        self.checkpoint_id = checkpoint_id


# Token Budget Errors
class TokenBudgetError(OrchestratorError):
    """Base class for token budget errors."""
    pass


class InsufficientTokensError(TokenBudgetError):
    """Raised when there are insufficient tokens for an operation."""
    
    def __init__(self, message="Insufficient tokens", requested=None, available=None):
        super().__init__(message, {
            'requested': requested,
            'available': available
        })
        self.requested = requested
        self.available = available


class BudgetExceededError(TokenBudgetError):
    """Raised when token budget is exceeded."""
    
    def __init__(self, message="Budget exceeded", used=None, budget=None):
        super().__init__(message, {
            'used': used,
            'budget': budget
        })
        self.used = used
        self.budget = budget


# Climate Justice Errors
class ClimateJusticeError(OrchestratorError):
    """Base class for climate justice errors."""
    pass


class JusticePricingError(ClimateJusticeError):
    """Raised when climate justice pricing calculation fails."""
    
    def __init__(self, message="Justice pricing error", region=None, purpose=None):
        super().__init__(message, {
            'region': region,
            'purpose': purpose
        })
        self.region = region
        self.purpose = purpose


class EmergencyResponseError(ClimateJusticeError):
    """Raised when climate emergency response fails."""
    
    def __init__(self, message="Emergency response failed", emergency_id=None):
        super().__init__(message, {'emergency_id': emergency_id})
        self.emergency_id = emergency_id


# Configuration Errors
class ConfigurationError(OrchestratorError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message="Configuration error", config_key=None):
        super().__init__(message, {'config_key': config_key})
        self.config_key = config_key


# Validation Errors  
class ValidationError(OrchestratorError):
    """Raised when input validation fails."""
    
    def __init__(self, message="Validation error", field=None, value=None):
        super().__init__(message, {
            'field': field,
            'value': value
        })
        self.field = field
        self.value = value


# Service Errors
class ServiceUnavailableError(OrchestratorError):
    """Raised when a required service is unavailable."""
    
    def __init__(self, message="Service unavailable", service_name=None):
        super().__init__(message, {'service_name': service_name})
        self.service_name = service_name


class CircuitBreakerError(OrchestratorError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message="Circuit breaker open", service_name=None, retry_after=None):
        super().__init__(message, {
            'service_name': service_name,
            'retry_after': retry_after
        })
        self.service_name = service_name
        self.retry_after = retry_after


# Utility functions for error handling
def handle_llm_error(error: Exception) -> LLMError:
    """Convert generic exceptions to specific LLM errors."""
    error_str = str(error).lower()
    
    if "context_length_exceeded" in error_str or "token limit" in error_str:
        return ContextWindowExhaustionError(str(error))
    elif "rate limit" in error_str or "too many requests" in error_str:
        return LLMTrafficError(str(error))
    elif "timeout" in error_str:
        return LLMTimeoutError(str(error))
    else:
        return LLMError(str(error))


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    retryable_errors = (
        LLMTrafficError,
        LLMTimeoutError,
        ClimateAPIError,
        ServiceUnavailableError
    )
    
    return isinstance(error, retryable_errors)


def get_retry_delay(error: Exception, attempt: int) -> float:
    """Get retry delay based on error type and attempt number."""
    base_delay = 1.0
    max_delay = 30.0
    
    if isinstance(error, LLMTrafficError) and hasattr(error, 'retry_after'):
        return float(error.retry_after or base_delay)
    
    # Exponential backoff with jitter
    import random
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0.1, 0.3) * delay
    
    return delay + jitter


# Error formatting for API responses
def format_error_response(error: Exception) -> dict:
    """Format error for JSON API response."""
    base_response = {
        'success': False,
        'error_type': type(error).__name__,
        'message': str(error)
    }
    
    # Add additional details if available
    if isinstance(error, OrchestratorError) and error.details:
        base_response['details'] = error.details
    
    # Add retry information for retryable errors
    if is_retryable_error(error):
        base_response['retryable'] = True
        base_response['retry_after'] = get_retry_delay(error, 0)
    else:
        base_response['retryable'] = False
    
    return base_response


if __name__ == "__main__":
    # Test error handling utilities
    print("🚨 Testing error handling utilities with Pydantic config")
    
    # Test error conversion
    test_errors = [
        Exception("context_length_exceeded: Token limit reached"),
        Exception("rate limit exceeded"),
        Exception("request timeout"),
        Exception("unknown error")
    ]
    
    for error in test_errors:
        converted = handle_llm_error(error)
        formatted = format_error_response(converted)
        print(f"Original: {type(error).__name__}")
        print(f"Converted: {type(converted).__name__}")
        print(f"API Response: {formatted}")
        print()
    
    # Test retry logic
    for attempt in range(3):
        delay = get_retry_delay(LLMTrafficError("Rate limited"), attempt)
        print(f"Attempt {attempt + 1}: Retry delay {delay:.2f}s")

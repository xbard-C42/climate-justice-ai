# Earth Justice AI Orchestrator

A Python package that **embeds climate justice and economic fairness** directly into every AI call. Rather than optional ethics checks, this library:

1. **Calculates a justice-aware price** for each request based on utility theory ("€200 k plateau" physics), cost-of-living, purpose, usage patterns, and verification status.
2. **Selects the fairest LLM provider** (OpenAI, Anthropic, Gemini, or a local model) according to region, climate priorities, and emergency levels.
3. **Prepend a transparent pricing breakdown** to every prompt, so users see exactly how their price was computed.

![Effect Estimation Framework UI]((https://github.com/xbard-C42/climate-justice-ai/blob/main/climate-justice-ai.jpeg))

## 🚀 Features

* **Marginal-Utility Pricing Engine**

  * Four economic tiers: `basic_needs`, `stable_living`, `comfortable`, `surplus_wealth`
  * Purpose-based multipliers (education, HFT, disaster response, etc.)
  * Regional cost-of-living adjustments and verification discounts/premiums
  * Climate discounts and redistributable surplus for social good

* **Justice-Aware Provider Routing**

  * Emergency overrides always pick the most reliable model (OpenAI)
  * Global South + climate work → Anthropic → OpenAI → Gemini fallback
  * Global South non-climate → Local → Gemini → OpenAI
  * Global North climate → OpenAI → Anthropic
  * Default commercial routing for other use cases

* **Single FastAPI Endpoint**

  * `POST /v1/ai/process` takes a JSON payload (region, purpose, emergency level, prompt) and returns the model’s response plus usage metrics.

## 📦 Installation

```bash
git clone https://github.com/your-org/earth-justice-orchestrator.git
cd earth-justice-orchestrator
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file with your API keys and desired settings:

```dotenv
OPENAI_API_KEY=sk-xxxx
ANTHROPIC_API_KEY=api-key-yyyy
gemini_api_key=...
LOCAL_MODEL_PATH=/path/to/model
```

Configure economic parameters via environment or a Pydantic settings file:

```yaml
utility_thresholds:
  basic_needs_max: 30000
  stable_living_max: 80000
  comfortable_max: 200000
tier_multipliers:
  basic_needs: 0.2
  stable_living: 0.7
  comfortable: 1.2
  surplus_wealth: 5.0
purpose_adjustments:
  education: 0.3
  high_frequency_trading: 6.0
  disaster_response: 0.5
cost_of_living:
  BD: 0.3
  US: 1.0
  GB: 0.9
verification_discount: 0.8
```

## 🚀 Usage

Run the FastAPI server:

```bash
uvicorn earth_justice.api:app --reload
```

Send a POST request:

```bash
curl -X POST http://localhost:8000/v1/ai/process \
  -H 'Content-Type: application/json' \
  -d '{
    "origin_region": "BD",
    "purpose": "climate_adaptation",
    "emergency_level": 2,
    "prompt": [{"role": "user", "content": "Predict flood risk in Dhaka tomorrow."}],
    "model": "gpt-4"
}'
```

Response example:

```json
{
  "content": "<AI-generated text>",
  "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
}
```

## 🏗️ Architecture

```
earth_justice/
├── __init__.py       # Exports
├── providers.py      # Provider enum & UnifiedLLMClient
├── economics.py      # EconomicContext & UtilityBasedPricingEngine
├── orchestrator.py   # EarthJusticeOrchestrator core logic
├── api.py            # FastAPI server
└── types.py          # Pydantic models & error classes
```

* **`providers.py`**: Detects configured LLM APIs and dispatches calls with unified error mapping.
* **`economics.py`**: Implements the full justice-pricing algorithm (marginal utility, region, purpose).
* **`orchestrator.py`**: Selects provider, builds transparent prompt, reserves budget, and executes the call.
* **`api.py`**: Exposes a simple REST endpoint.

## ✅ Testing

Run the unit and smoke tests:

```bash
pytest tests/
```

## 🤝 Contribute

We welcome improvements! Please open issues or pull requests for:

* New purpose categories or region mappings
* Additional provider integrations (e.g., local LLMs)
* Monitoring, metrics, and CI/CD enhancements

---

*Earth Justice AI Orchestrator* turns fairness from a buzzword into code — because helping those who need it most shouldn’t be optional.



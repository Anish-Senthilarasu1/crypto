"""
DeepSeek R1 Trading Agent using Ollama
Autonomous AI decision making for crypto trading
"""

import json
import logging
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class DeepSeekTradingAgent:
    """AI trading agent powered by DeepSeek R1 via Ollama"""

    def __init__(self, model: str = "deepseek-r1:14b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

    async def test_connection(self):
        """Test Ollama connection and model availability"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            models = response.json()

            model_names = [m['name'] for m in models.get('models', [])]
            if self.model not in model_names:
                logger.warning(f"⚠️ Model {self.model} not found. Available: {model_names}")
                logger.info(f"💡 Run: ollama pull {self.model}")
            else:
                logger.info(f"✅ Ollama connected. Model {self.model} ready.")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise

    async def make_trading_decision(
        self,
        market_data: Dict[str, Any],
        signals: Dict[str, Any],
        regime: str,
        account: Dict[str, Any],
        positions: list
    ) -> Dict[str, Any]:
        """
        Use DeepSeek R1 to make autonomous trading decisions

        Returns:
            {
                'action': 'buy' | 'sell' | 'hold',
                'symbol': str,
                'quantity': float,
                'reasoning': str,
                'confidence': float
            }
        """

        # Construct detailed prompt for DeepSeek
        prompt = self._build_trading_prompt(market_data, signals, regime, account, positions)

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower for more deterministic trading
                        "top_p": 0.9,
                    }
                }
            )

            result = response.json()
            ai_response = result['response']

            # Parse AI response into structured decision
            decision = self._parse_decision(ai_response)

            logger.info(f"🧠 AI Reasoning: {decision.get('reasoning', 'N/A')}")

            return decision

        except Exception as e:
            logger.error(f"❌ Error getting AI decision: {e}")
            return {
                'action': 'hold',
                'symbol': None,
                'quantity': 0,
                'reasoning': f'Error: {str(e)}',
                'confidence': 0.0
            }

    def _build_trading_prompt(
        self,
        market_data: Dict[str, Any],
        signals: Dict[str, Any],
        regime: str,
        account: Dict[str, Any],
        positions: list
    ) -> str:
        """Build comprehensive trading prompt for DeepSeek"""

        prompt = f"""You are an expert cryptocurrency trading AI agent. Analyze the following data and make a trading decision.

## Current Market Data
{json.dumps(market_data, indent=2)}

## Strategy Signals
{json.dumps(signals, indent=2)}

## Market Regime
{regime}

## Account Status
- Buying Power: ${account.get('buying_power', 0):,.2f}
- Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
- Cash: ${account.get('cash', 0):,.2f}

## Current Positions
{json.dumps(positions, indent=2)}

## Trading Strategies Available
1. **Intraday Momentum**: Exploits first-half-hour to last-half-hour correlation (Sharpe 1.72)
2. **Copula Mean Reversion**: High-frequency mean reversion on 5-min data (Sharpe 3.77)
3. **Multi-Level DQN**: Sentiment-enhanced deep RL strategy (Sharpe 2.74)

## Your Task
Based on the mathematically-backed strategies and current market conditions:

1. Decide: BUY, SELL, or HOLD
2. If BUY/SELL, specify symbol and quantity
3. Provide clear reasoning referencing the strategy signals
4. Assign confidence level (0.0 to 1.0)

## Risk Management Rules
- Maximum position size: 10% of portfolio per trade
- Use fractional Kelly criterion (0.25-0.5)
- Stop loss: 5% for mean reversion positions
- Only trade during high-volume, high-volatility periods
- Respect regime-based allocations

Respond in this exact JSON format:
{{
  "action": "buy|sell|hold",
  "symbol": "BTC/USD",
  "quantity": 0.0,
  "reasoning": "detailed explanation",
  "confidence": 0.85,
  "strategy_used": "momentum|mean_reversion|dqn|hybrid"
}}
"""
        return prompt

    def _parse_decision(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into structured decision"""

        try:
            # Try to extract JSON from response
            # DeepSeek might wrap it in markdown code blocks
            if "```json" in ai_response:
                json_str = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                json_str = ai_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = ai_response

            decision = json.loads(json_str)

            # Validate required fields
            required_fields = ['action', 'symbol', 'quantity', 'reasoning', 'confidence']
            if not all(field in decision for field in required_fields):
                raise ValueError("Missing required fields in decision")

            # Validate action
            if decision['action'] not in ['buy', 'sell', 'hold']:
                decision['action'] = 'hold'

            return decision

        except Exception as e:
            logger.error(f"❌ Error parsing AI response: {e}")
            logger.debug(f"Raw response: {ai_response}")

            # Default to hold on parse error
            return {
                'action': 'hold',
                'symbol': None,
                'quantity': 0,
                'reasoning': f'Failed to parse AI response: {str(e)}',
                'confidence': 0.0,
                'strategy_used': 'none'
            }

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

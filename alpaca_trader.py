"""
Alpaca API Integration for Crypto Trading
Supports real-time crypto trading via Alpaca
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import httpx
import asyncio

logger = logging.getLogger(__name__)


class AlpacaCryptoTrader:
    """Alpaca crypto trading client"""

    def __init__(self):
        # Get credentials from environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

        # Set base URLs
        if self.paper:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"

        # HTTP client with auth headers
        self.client = httpx.AsyncClient(
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            },
            timeout=30.0
        )

        logger.info(f"🔑 Alpaca initialized ({'PAPER' if self.paper else 'LIVE'} trading)")

    async def test_connection(self):
        """Test Alpaca API connection"""
        try:
            account = await self.get_account()
            logger.info(f"✅ Alpaca connected. Portfolio: ${account['portfolio_value']}")
            logger.info(f"💵 Buying Power: ${account['buying_power']}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alpaca: {e}")
            raise

    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        response = await self.client.get(f"{self.base_url}/v2/account")
        response.raise_for_status()

        data = response.json()
        return {
            'buying_power': float(data['buying_power']),
            'cash': float(data['cash']),
            'portfolio_value': float(data['portfolio_value']),
            'equity': float(data['equity']),
            'pattern_day_trader': data.get('pattern_day_trader', False)
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        response = await self.client.get(f"{self.base_url}/v2/positions")
        response.raise_for_status()

        positions = []
        for pos in response.json():
            positions.append({
                'symbol': pos['symbol'],
                'qty': float(pos['qty']),
                'avg_entry_price': float(pos['avg_entry_price']),
                'current_price': float(pos['current_price']),
                'market_value': float(pos['market_value']),
                'unrealized_pl': float(pos['unrealized_pl']),
                'unrealized_plpc': float(pos['unrealized_plpc'])
            })

        return positions

    async def get_market_data(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get real-time market data for crypto symbols

        Args:
            symbols: List of symbols (e.g., ['BTC/USD', 'ETH/USD'])
        """
        if symbols is None:
            symbols = ['BTC/USD', 'ETH/USD']  # Default crypto pairs

        market_data = {}

        for symbol in symbols:
            try:
                # Get latest trade
                trade_response = await self.client.get(
                    f"{self.data_url}/v1beta3/crypto/us/latest/trades",
                    params={'symbols': symbol}
                )
                trade_response.raise_for_status()
                trade_data = trade_response.json()

                # Get latest quote
                quote_response = await self.client.get(
                    f"{self.data_url}/v1beta3/crypto/us/latest/quotes",
                    params={'symbols': symbol}
                )
                quote_response.raise_for_status()
                quote_data = quote_response.json()

                # Get bars (OHLCV) for technical analysis
                bars_response = await self.client.get(
                    f"{self.data_url}/v1beta3/crypto/us/bars",
                    params={
                        'symbols': symbol,
                        'timeframe': '1Min',
                        'limit': 100
                    }
                )
                bars_response.raise_for_status()
                bars_data = bars_response.json()

                market_data[symbol] = {
                    'trade': trade_data.get('trades', {}).get(symbol, {}),
                    'quote': quote_data.get('quotes', {}).get(symbol, {}),
                    'bars': bars_data.get('bars', {}).get(symbol, []),
                    'timestamp': datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
                market_data[symbol] = None

        return market_data

    async def execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade based on AI decision

        Args:
            decision: {
                'action': 'buy' | 'sell',
                'symbol': 'BTC/USD',
                'quantity': float,
                'reasoning': str
            }
        """
        action = decision['action']
        symbol = decision['symbol']
        quantity = decision['quantity']

        if action == 'hold':
            return {'status': 'hold', 'message': 'No action taken'}

        try:
            # Determine order side
            side = 'buy' if action == 'buy' else 'sell'

            # Create market order
            order_payload = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': 'market',
                'time_in_force': 'gtc'  # Good till cancel
            }

            logger.info(f"📤 Submitting {side.upper()} order: {quantity} {symbol}")

            response = await self.client.post(
                f"{self.base_url}/v2/orders",
                json=order_payload
            )
            response.raise_for_status()

            order = response.json()

            result = {
                'status': 'submitted',
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'submitted_at': order['submitted_at'],
                'reasoning': decision.get('reasoning', '')
            }

            logger.info(f"✅ Order submitted: {order['id']}")

            return result

        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'decision': decision
            }

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status by ID"""
        response = await self.client.get(f"{self.base_url}/v2/orders/{order_id}")
        response.raise_for_status()
        return response.json()

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            response = await self.client.delete(f"{self.base_url}/v2/orders/{order_id}")
            response.raise_for_status()
            logger.info(f"✅ Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"❌ Error cancelling order: {e}")
            return False

    async def close_all_positions(self):
        """Close all open positions"""
        try:
            response = await self.client.delete(f"{self.base_url}/v2/positions")
            response.raise_for_status()
            logger.info("✅ All positions closed")
            return response.json()
        except Exception as e:
            logger.error(f"❌ Error closing positions: {e}")
            return None

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = '1Min',
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV bars

        Args:
            symbol: Crypto pair (e.g., 'BTC/USD')
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start: Start datetime
            end: End datetime
            limit: Max number of bars
        """
        params = {
            'symbols': symbol,
            'timeframe': timeframe,
            'limit': limit
        }

        if start:
            params['start'] = start.isoformat()
        if end:
            params['end'] = end.isoformat()

        response = await self.client.get(
            f"{self.data_url}/v1beta3/crypto/us/bars",
            params=params
        )
        response.raise_for_status()

        data = response.json()
        bars = data.get('bars', {}).get(symbol, [])

        return bars

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

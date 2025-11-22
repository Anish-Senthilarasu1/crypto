#!/usr/bin/env python3
"""
Autonomous Crypto Trading Bot with Ollama DeepSeek R1 14B
Implements mathematically-backed strategies with AI-driven decision making
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any
import signal
import sys

# Create logs directory BEFORE logging configuration
os.makedirs("logs", exist_ok=True)
os.makedirs("static", exist_ok=True)

from trading_agent import DeepSeekTradingAgent
from alpaca_trader import AlpacaCryptoTrader
from strategy_engine import StrategyEngine
from monitoring import PerformanceMonitor
from dashboard import dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class CryptoTradingBot:
    """Main orchestrator for autonomous crypto trading"""

    def __init__(self):
        self.agent = DeepSeekTradingAgent(model="deepseek-r1:14b")
        self.trader = AlpacaCryptoTrader()
        self.strategy_engine = StrategyEngine()
        self.monitor = PerformanceMonitor()
        self.running = False

    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Crypto Trading Bot...")

        # Start dashboard server
        await dashboard.start()

        # Test Ollama connection
        await self.agent.test_connection()

        # Test Alpaca connection
        await self.trader.test_connection()

        # Load historical data for strategy calibration
        await self.strategy_engine.initialize()

        logger.info("✅ All systems initialized successfully")

    async def run_trading_loop(self):
        """Main 24/7 trading loop"""
        self.running = True
        logger.info("🔄 Starting 24/7 trading loop...")

        while self.running:
            try:
                # Get current market data
                market_data = await self.trader.get_market_data()

                # Get account status
                account = await self.trader.get_account()
                positions = await self.trader.get_positions()

                # Update dashboard with account info
                await dashboard.update_account(account)
                await dashboard.update_positions(positions)

                # Detect market regime
                regime = await self.strategy_engine.detect_regime(market_data)
                logger.info(f"📊 Current market regime: {regime}")

                # Generate strategy signals
                signals = await self.strategy_engine.generate_signals(
                    market_data, regime
                )

                # Update dashboard with signals
                await dashboard.update_signals(signals)

                # Get AI agent decision
                decision = await self.agent.make_trading_decision(
                    market_data=market_data,
                    signals=signals,
                    regime=regime,
                    account=account,
                    positions=positions
                )

                logger.info(f"🤖 DeepSeek Decision: {decision}")

                # Update dashboard with AI reasoning
                await dashboard.add_ai_reasoning(decision)

                # Execute trades based on decision
                if decision['action'] != 'hold':
                    execution_result = await self.trader.execute_trade(decision)
                    logger.info(f"💰 Trade executed: {execution_result}")

                    # Update monitoring
                    await self.monitor.record_trade(execution_result)

                    # Update dashboard with trade
                    await dashboard.add_trade(execution_result)

                # Log performance metrics
                await self.monitor.log_metrics(account, positions)

                # Calculate and update metrics on dashboard
                sharpe = self.monitor.calculate_sharpe_ratio(30)
                max_dd = self.monitor.calculate_max_drawdown()
                await dashboard.update_metrics({
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'total_trades': len(self.monitor.trades)
                })

                # Sleep between iterations (adjust based on strategy frequency)
                await asyncio.sleep(60)  # 1 minute for high-frequency

            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down trading bot...")
        self.running = False

        # Close all positions if configured
        # asyncio.create_task(self.trader.close_all_positions())

        sys.exit(0)


async def main():
    """Main entry point"""
    bot = CryptoTradingBot()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, bot.shutdown)
    signal.signal(signal.SIGTERM, bot.shutdown)

    try:
        await bot.initialize()
        await bot.run_trading_loop()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

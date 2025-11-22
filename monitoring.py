"""
Performance Monitoring and Logging System
Tracks trades, calculates metrics, and generates reports
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors and logs trading performance"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.trades_file = self.log_dir / "trades.jsonl"
        self.metrics_file = self.log_dir / "metrics.jsonl"

        self.trades = []
        self.metrics_history = []

    async def record_trade(self, execution_result: Dict[str, Any]):
        """Record a trade execution"""

        trade_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': execution_result.get('status'),
            'order_id': execution_result.get('order_id'),
            'symbol': execution_result.get('symbol'),
            'side': execution_result.get('side'),
            'quantity': execution_result.get('quantity'),
            'reasoning': execution_result.get('reasoning', ''),
            'submitted_at': execution_result.get('submitted_at')
        }

        self.trades.append(trade_record)

        # Append to JSONL file
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(trade_record) + '\n')

        logger.info(f"📝 Trade recorded: {trade_record['symbol']} {trade_record['side']}")

    async def log_metrics(self, account: Dict[str, Any], positions: List[Dict[str, Any]]):
        """Log current performance metrics"""

        # Calculate total unrealized P&L
        total_unrealized_pl = sum(p.get('unrealized_pl', 0) for p in positions)
        total_unrealized_plpc = sum(p.get('unrealized_plpc', 0) for p in positions) / max(len(positions), 1)

        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'portfolio_value': account['portfolio_value'],
            'cash': account['cash'],
            'buying_power': account['buying_power'],
            'equity': account['equity'],
            'num_positions': len(positions),
            'total_unrealized_pl': total_unrealized_pl,
            'total_unrealized_plpc': total_unrealized_plpc,
            'positions': positions
        }

        self.metrics_history.append(metrics)

        # Log every hour to file
        if len(self.metrics_history) % 60 == 0:  # Every 60 iterations (if 1-min loop)
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        # Console log
        logger.info(f"💰 Portfolio: ${metrics['portfolio_value']:,.2f} | "
                   f"P&L: ${total_unrealized_pl:+,.2f} ({total_unrealized_plpc:+.2%})")

    def calculate_sharpe_ratio(self, window_days: int = 30) -> float:
        """
        Calculate rolling Sharpe ratio from metrics history

        Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
        """

        if len(self.metrics_history) < 2:
            return 0.0

        try:
            # Get recent metrics
            recent_metrics = self.metrics_history[-window_days * 24 * 60:]  # Assuming 1-min data

            portfolio_values = [m['portfolio_value'] for m in recent_metrics]

            # Calculate returns
            import numpy as np
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            if len(returns) < 2:
                return 0.0

            # Annualized Sharpe (assuming risk-free rate = 0 for crypto)
            mean_return = np.mean(returns) * 365 * 24 * 60  # Annualized
            std_return = np.std(returns) * np.sqrt(365 * 24 * 60)  # Annualized

            if std_return == 0:
                return 0.0

            sharpe = mean_return / std_return

            return sharpe

        except Exception as e:
            logger.error(f"❌ Error calculating Sharpe: {e}")
            return 0.0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak"""

        if len(self.metrics_history) < 2:
            return 0.0

        try:
            import numpy as np

            portfolio_values = [m['portfolio_value'] for m in self.metrics_history]
            portfolio_values = np.array(portfolio_values)

            # Calculate running maximum
            running_max = np.maximum.accumulate(portfolio_values)

            # Calculate drawdown
            drawdown = (portfolio_values - running_max) / running_max

            max_dd = np.min(drawdown)

            return max_dd

        except Exception as e:
            logger.error(f"❌ Error calculating max drawdown: {e}")
            return 0.0

    def generate_daily_report(self):
        """Generate daily performance report"""

        logger.info("=" * 60)
        logger.info("📊 DAILY PERFORMANCE REPORT")
        logger.info("=" * 60)

        # Total trades today
        from datetime import date
        today_trades = [t for t in self.trades if t['timestamp'].startswith(date.today().isoformat())]

        logger.info(f"Total trades today: {len(today_trades)}")

        # Current metrics
        if self.metrics_history:
            latest = self.metrics_history[-1]
            logger.info(f"Portfolio Value: ${latest['portfolio_value']:,.2f}")
            logger.info(f"Cash: ${latest['cash']:,.2f}")
            logger.info(f"Total P&L: ${latest['total_unrealized_pl']:+,.2f}")

        # Calculate Sharpe
        sharpe_30d = self.calculate_sharpe_ratio(30)
        logger.info(f"30-Day Sharpe Ratio: {sharpe_30d:.2f}")

        # Max Drawdown
        max_dd = self.calculate_max_drawdown()
        logger.info(f"Max Drawdown: {max_dd:.2%}")

        logger.info("=" * 60)

"""
Configuration Management
Loads environment variables and provides configuration
"""

import os
from typing import List
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Application configuration"""

    # Alpaca API
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
    ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

    # Ollama
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-r1:14b')

    # Trading
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.10'))
    RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.0'))
    TRADING_SYMBOLS = os.getenv('TRADING_SYMBOLS', 'BTC/USD,ETH/USD').split(',')

    # Risk Management
    MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.30'))
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.05'))
    KELLY_FRACTION = float(os.getenv('KELLY_FRACTION', '0.25'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('LOG_DIR', 'logs')

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.ALPACA_API_KEY or not cls.ALPACA_API_SECRET:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")

        if cls.ALPACA_API_KEY == "your_alpaca_api_key_here":
            raise ValueError("Please set real Alpaca API credentials in .env file")

        return True


# Validate on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️ Configuration warning: {e}")
    print("💡 Copy .env.example to .env and add your credentials")

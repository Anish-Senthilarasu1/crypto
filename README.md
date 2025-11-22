# 🤖 Autonomous Crypto Trading Bot

**AI-Powered 24/7 Cryptocurrency Trading using DeepSeek R1 (Ollama) + Alpaca**

This trading bot implements three mathematically-backed strategies with AI-driven decision making using DeepSeek R1 14B running locally on your gaming PC.

## 🎯 Features

- **🎨 Premium Real-Time Dashboard**: Beautiful web UI with live trading visualization
- **🤖 DeepSeek R1 14B AI Agent**: Local LLM inference via Ollama for autonomous trading decisions
- **💹 Alpaca Crypto Trading**: Real-time crypto trading with commission-free execution
- **📊 Three Proven Strategies**:
  - 📈 **Intraday Momentum** (Sharpe 1.72, 16.69% annual returns)
  - 🔄 **Copula Mean Reversion** (Sharpe 3.77, 75.2% annual returns)
  - 🧠 **Multi-Level DQN** (Sharpe 2.74, 29.93% ROI)
- **🎯 Regime Detection**: Automatically adapts strategy allocation to market conditions
- **🛡️ Risk Management**: Kelly Criterion, stop-losses, max drawdown controls
- **⚡ 24/7 Operation**: Autonomous trading with monitoring and logging

## 📋 Prerequisites

### 1. Gaming PC Requirements
- **GPU**: NVIDIA RTX 3060+ (or equivalent with 8GB+ VRAM)
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 20GB+ free space
- **OS**: Windows/Linux/macOS

### 2. Ollama Installation

Install Ollama from: https://ollama.com/

```bash
# Download and install Ollama
# Then pull DeepSeek R1 14B model
ollama pull deepseek-r1:14b
```

Verify installation:
```bash
ollama list
# Should show: deepseek-r1:14b
```

### 3. Alpaca Account

1. Sign up at: https://alpaca.markets/
2. Get API keys from dashboard
3. Start with **Paper Trading** (free, no real money)

## 🚀 Quick Start

### 1. Clone and Install

```bash
cd /Users/anishsenthilarasu/Desktop/crypto_local

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Add your Alpaca API credentials:
```env
ALPACA_API_KEY=your_actual_key_here
ALPACA_API_SECRET=your_actual_secret_here
ALPACA_PAPER=true  # Keep true for testing!
```

### 3. Run the Bot

```bash
python main.py
```

You should see:
```
🚀 Initializing Crypto Trading Bot...
🎨 Dashboard running at http://0.0.0.0:8080
🌐 Open in browser: http://localhost:8080
✅ Ollama connected. Model deepseek-r1:14b ready.
✅ Alpaca connected. Portfolio: $100000.00
💵 Buying Power: $100000.00
✅ All systems initialized successfully
🔄 Starting 24/7 trading loop...
```

### 4. Open the Dashboard

Open your browser and go to:
```
http://localhost:8080
```

You'll see a **premium real-time dashboard** featuring:
- 💰 Live portfolio value and P&L
- 📈 Interactive performance charts
- 🤖 AI decision reasoning in real-time
- 📊 Strategy signals and regime detection
- 💼 Current positions with live P&L
- 📝 Recent trades log
- 🎯 Sharpe ratio and performance metrics

The dashboard auto-updates in real-time via WebSocket!

## 📊 Strategy Overview

### Intraday Momentum
- **How it works**: Exploits correlation between first-half-hour and last-half-hour returns
- **Best for**: High-volume, high-volatility bull markets
- **Risk**: Requires 10:1 leverage; use carefully

### Copula Mean Reversion
- **How it works**: Identifies mean-reverting price relationships using statistical cointegration
- **Best for**: Ranging, sideways markets
- **Risk**: Breaks down during strong trends

### AI Decision Making
The DeepSeek R1 agent:
1. Receives strategy signals and market data
2. Analyzes regime and risk conditions
3. Makes autonomous buy/sell/hold decisions
4. Provides reasoning for each trade

## 🛡️ Risk Management

Built-in safeguards:
- **Max Position Size**: 10% of portfolio per trade
- **Stop Loss**: 5% on mean reversion positions
- **Max Drawdown**: 30% triggers shutdown
- **Kelly Criterion**: Quarter-Kelly (0.25) for position sizing
- **Paper Trading**: Test without real money

## 📈 Monitoring

Logs are saved in `logs/` directory:
- `trading_bot_YYYYMMDD.log`: Detailed activity log
- `trades.jsonl`: All executed trades
- `metrics.jsonl`: Hourly performance metrics

View real-time performance:
```bash
tail -f logs/trading_bot_$(date +%Y%m%d).log
```

## 🔧 Advanced Configuration

Edit [config.py](config.py) or `.env` for:
- Trading symbols (default: BTC/USD, ETH/USD)
- Risk parameters (Kelly fraction, stop-loss %)
- Strategy allocation weights
- Logging levels

## ⚠️ Important Warnings

1. **Start with Paper Trading**: Always test with paper trading first
2. **Crypto is Volatile**: 50-80% annual volatility is normal
3. **Past Performance ≠ Future Results**: Strategies can fail
4. **Monitor Regularly**: Check logs and metrics daily
5. **Risk Capital Only**: Never trade money you can't afford to lose

## 🐛 Troubleshooting

### Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# On Mac/Linux: killall ollama && ollama serve
# On Windows: Restart Ollama app
```

### Alpaca API Errors
- Verify API keys in `.env`
- Check paper trading mode: `ALPACA_PAPER=true`
- Ensure crypto trading is enabled in Alpaca account

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Resources

- **Strategy Research**: See [strategy.md](strategy.md) for full mathematical details
- **Ollama Docs**: https://github.com/ollama/ollama
- **Alpaca Crypto API**: https://docs.alpaca.markets/docs/crypto-trading
- **DeepSeek Models**: https://ollama.com/library/deepseek-r1

## 🤝 Contributing

This is a personal trading bot. Use and modify at your own risk.

## 📄 License

MIT License - Use at your own risk. No warranty provided.

## ⚡ Performance Tips

**GPU Optimization**:
- Close other GPU-intensive apps while trading
- Monitor GPU memory: `nvidia-smi` (Linux/Windows)
- Reduce temperature for 24/7 operation

**Network**:
- Use stable, low-latency internet
- Consider running on cloud GPU if local GPU is insufficient

**Strategy Tuning**:
- Start conservative (small positions)
- Monitor Sharpe ratio and drawdown
- Adjust regime detection thresholds based on market

## 🎮 Gaming PC Deployment

Your gaming PC is perfect for this because:
- ✅ Powerful GPU for local LLM inference
- ✅ Can run 24/7 in the background
- ✅ No cloud costs (free after setup)
- ✅ Full control and privacy

Just ensure:
- Stable power supply (UPS recommended)
- Good cooling for extended operation
- Backup system for critical data

---

**Happy Trading! 🚀💰**

Remember: This is experimental software. Always start with paper trading and never risk more than you can afford to lose.

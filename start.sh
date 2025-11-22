#!/bin/bash
# Quick start script for the crypto trading bot

echo "🚀 Starting Crypto Trading Bot..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "✋ IMPORTANT: Edit .env file with your Alpaca API credentials"
    echo "   Then run this script again."
    exit 1
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running!"
    echo "💡 Start Ollama first: ollama serve"
    echo ""
    read -p "   Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Check if deepseek model is pulled
if ! ollama list | grep -q "deepseek-r1"; then
    echo "📥 DeepSeek R1 model not found. Pulling now..."
    ollama pull deepseek-r1:14b
fi

# Create logs directory
mkdir -p logs
mkdir -p static

echo ""
echo "✅ Starting trading bot..."
echo "🌐 Dashboard will be available at: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the bot"
echo ""

# Run the bot
python main.py

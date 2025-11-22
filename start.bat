@echo off
REM Quick start script for Windows

echo 🚀 Starting Crypto Trading Bot...
echo.

REM Check if .env exists
if not exist .env (
    echo ⚠️  No .env file found!
    echo 📝 Creating .env from template...
    copy .env.example .env
    echo.
    echo ✋ IMPORTANT: Edit .env file with your Alpaca API credentials
    echo    Then run this script again.
    pause
    exit /b 1
)

REM Create directories
if not exist logs mkdir logs
if not exist static mkdir static

echo.
echo ✅ Starting trading bot...
echo 🌐 Dashboard will be available at: http://localhost:8080
echo.
echo Press Ctrl+C to stop the bot
echo.

REM Run the bot
python main.py

"""
Premium Real-Time Trading Dashboard
Beautiful web UI for monitoring the AI trading agent
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from aiohttp import web
import aiohttp_cors

logger = logging.getLogger(__name__)


class TradingDashboard:
    """WebSocket-based real-time dashboard"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websockets = set()
        self.latest_data = {
            'account': {},
            'positions': [],
            'signals': {},
            'trades': [],
            'metrics': {},
            'ai_reasoning': []
        }

    def setup_routes(self):
        """Setup HTTP and WebSocket routes"""

        # Serve static files (HTML, CSS, JS)
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/api/data', self.api_data_handler)

        # Enable CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })

        for route in list(self.app.router.routes()):
            cors.add(route)

    async def index_handler(self, request):
        """Serve main dashboard HTML"""
        html_file = Path(__file__).parent / 'static' / 'index.html'
        return web.FileResponse(html_file)

    async def websocket_handler(self, request):
        """WebSocket connection for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.websockets.add(ws)
        logger.info(f"📡 New dashboard connection. Total: {len(self.websockets)}")

        # Send initial data
        await ws.send_json({
            'type': 'initial',
            'data': self.latest_data
        })

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle commands from dashboard
                    data = json.loads(msg.data)
                    await self.handle_dashboard_command(data, ws)

        finally:
            self.websockets.discard(ws)
            logger.info(f"📡 Dashboard disconnected. Total: {len(self.websockets)}")

        return ws

    async def api_data_handler(self, request):
        """REST API endpoint for current data"""
        return web.json_response(self.latest_data)

    async def handle_dashboard_command(self, data: Dict[str, Any], ws):
        """Handle commands from dashboard UI"""
        command = data.get('command')

        if command == 'pause_trading':
            logger.info("⏸️ Trading paused from dashboard")
            # Implement pause logic

        elif command == 'resume_trading':
            logger.info("▶️ Trading resumed from dashboard")
            # Implement resume logic

        elif command == 'close_all_positions':
            logger.warning("🛑 Close all positions requested from dashboard")
            # Implement close all logic

    async def broadcast_update(self, update_type: str, data: Any):
        """Broadcast update to all connected dashboards"""
        message = {
            'type': update_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Update latest data
        if update_type in self.latest_data:
            self.latest_data[update_type] = data

        # Broadcast to all WebSocket clients
        if self.websockets:
            await asyncio.gather(
                *[ws.send_json(message) for ws in self.websockets],
                return_exceptions=True
            )

    async def update_account(self, account: Dict[str, Any]):
        """Update account information"""
        await self.broadcast_update('account', account)

    async def update_positions(self, positions: List[Dict[str, Any]]):
        """Update positions"""
        await self.broadcast_update('positions', positions)

    async def update_signals(self, signals: Dict[str, Any]):
        """Update strategy signals"""
        await self.broadcast_update('signals', signals)

    async def add_trade(self, trade: Dict[str, Any]):
        """Add new trade to history"""
        self.latest_data['trades'].insert(0, trade)
        self.latest_data['trades'] = self.latest_data['trades'][:50]  # Keep last 50
        await self.broadcast_update('trade', trade)

    async def add_ai_reasoning(self, reasoning: Dict[str, Any]):
        """Add AI reasoning to display"""
        self.latest_data['ai_reasoning'].insert(0, reasoning)
        self.latest_data['ai_reasoning'] = self.latest_data['ai_reasoning'][:10]  # Keep last 10
        await self.broadcast_update('ai_reasoning', reasoning)

    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics"""
        await self.broadcast_update('metrics', metrics)

    async def start(self):
        """Start dashboard server"""
        self.setup_routes()

        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"🎨 Dashboard running at http://{self.host}:{self.port}")
        logger.info(f"🌐 Open in browser: http://localhost:{self.port}")


# Singleton instance
dashboard = TradingDashboard()

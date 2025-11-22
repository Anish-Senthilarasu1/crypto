// Premium Real-Time Trading Dashboard
// WebSocket connection and data visualization

let ws = null;
let portfolioChart = null;
let reconnectInterval = null;

// Chart colors
const CHART_COLORS = {
    gradient1: 'rgba(102, 126, 234, 0.3)',
    gradient2: 'rgba(118, 75, 162, 0.1)',
    line: '#667eea',
    grid: 'rgba(255, 255, 255, 0.1)'
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initPortfolioChart();
    connectWebSocket();
});

// WebSocket Connection
function connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname}:8080/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('🔗 Connected to trading bot');
        updateStatus('ACTIVE', true);
        if (reconnectInterval) {
            clearInterval(reconnectInterval);
            reconnectInterval = null;
        }
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        updateStatus('ERROR', false);
    };

    ws.onclose = () => {
        console.log('🔌 Disconnected from trading bot');
        updateStatus('DISCONNECTED', false);

        // Attempt reconnection every 5 seconds
        if (!reconnectInterval) {
            reconnectInterval = setInterval(() => {
                console.log('🔄 Attempting to reconnect...');
                connectWebSocket();
            }, 5000);
        }
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(message) {
    const { type, data } = message;

    switch (type) {
        case 'initial':
            handleInitialData(data);
            break;
        case 'account':
            updateAccountStats(data);
            break;
        case 'positions':
            updatePositions(data);
            break;
        case 'signals':
            updateSignals(data);
            break;
        case 'trade':
            addTrade(data);
            break;
        case 'ai_reasoning':
            addAIReasoning(data);
            break;
        case 'metrics':
            updateMetrics(data);
            break;
    }
}

// Handle initial data load
function handleInitialData(data) {
    if (data.account) updateAccountStats(data.account);
    if (data.positions) updatePositions(data.positions);
    if (data.signals) updateSignals(data.signals);
    if (data.trades) {
        data.trades.forEach(trade => addTrade(trade, false));
    }
    if (data.ai_reasoning) {
        data.ai_reasoning.forEach(reasoning => addAIReasoning(reasoning, false));
    }
    if (data.metrics) updateMetrics(data.metrics);
}

// Update account statistics
function updateAccountStats(account) {
    // Portfolio value
    const portfolioValue = account.portfolio_value || 0;
    document.getElementById('portfolio-value').textContent = formatCurrency(portfolioValue);

    // Calculate P&L (assuming initial balance was in account data or from history)
    const totalPL = account.equity - account.cash || 0;
    const totalPLPct = account.cash > 0 ? ((totalPL / account.cash) * 100) : 0;

    const plElement = document.getElementById('total-pl');
    const plChangeElement = document.getElementById('pl-change');

    plElement.textContent = formatCurrency(totalPL);
    plElement.style.color = totalPL >= 0 ? '#10b981' : '#ef4444';

    plChangeElement.className = `stat-change ${totalPL >= 0 ? 'positive' : 'negative'}`;
    plChangeElement.innerHTML = `
        <i class="fas fa-arrow-${totalPL >= 0 ? 'up' : 'down'}"></i>
        <span>${totalPLPct.toFixed(2)}%</span>
    `;

    // Update chart with new data point
    updatePortfolioChart(portfolioValue);
}

// Update positions table
function updatePositions(positions) {
    const tbody = document.getElementById('positions-table');
    const numPositionsEl = document.getElementById('num-positions');

    numPositionsEl.textContent = `${positions.length} position${positions.length !== 1 ? 's' : ''}`;

    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #8892b0;">No positions</td></tr>';
        return;
    }

    tbody.innerHTML = positions.map(pos => `
        <tr>
            <td><strong>${pos.symbol}</strong></td>
            <td>${pos.qty}</td>
            <td>${formatCurrency(pos.avg_entry_price)}</td>
            <td>${formatCurrency(pos.current_price)}</td>
            <td class="${pos.unrealized_pl >= 0 ? 'positive-pl' : 'negative-pl'}">
                ${formatCurrency(pos.unrealized_pl)} (${(pos.unrealized_plpc * 100).toFixed(2)}%)
            </td>
        </tr>
    `).join('');
}

// Update strategy signals
function updateSignals(signals) {
    // Momentum
    if (signals.momentum) {
        const momentumSignal = signals.momentum.signal || 'hold';
        const momentumConfidence = (signals.momentum.confidence || 0) * 100;

        document.getElementById('momentum-signal').textContent = momentumSignal.toUpperCase();
        document.getElementById('momentum-signal').style.color = getSignalColor(momentumSignal);
        document.getElementById('momentum-strength').style.width = `${momentumConfidence}%`;
    }

    // Mean Reversion
    if (signals.mean_reversion) {
        const mrSignal = signals.mean_reversion.signal || 'hold';
        const mrConfidence = (signals.mean_reversion.confidence || 0) * 100;

        document.getElementById('mr-signal').textContent = mrSignal.toUpperCase();
        document.getElementById('mr-signal').style.color = getSignalColor(mrSignal);
        document.getElementById('mr-strength').style.width = `${mrConfidence}%`;
    }

    // Regime
    if (signals.regime) {
        const regimeEl = document.getElementById('regime');
        regimeEl.textContent = signals.regime.toUpperCase();
        regimeEl.style.color = getRegimeColor(signals.regime);
    }
}

// Add trade to recent trades list
function addTrade(trade, prepend = true) {
    const tradesListEl = document.getElementById('trades-list');

    const tradeHtml = `
        <div class="trade-item">
            <div>
                <span class="trade-side ${trade.side}">${trade.side?.toUpperCase() || 'N/A'}</span>
                <strong style="margin-left: 12px;">${trade.symbol}</strong>
            </div>
            <div style="text-align: right;">
                <div>${trade.quantity} @ ${formatCurrency(trade.price || 0)}</div>
                <div style="font-size: 11px; color: #8892b0;">
                    ${formatTimestamp(trade.submitted_at || trade.timestamp)}
                </div>
            </div>
        </div>
    `;

    if (prepend) {
        tradesListEl.insertAdjacentHTML('afterbegin', tradeHtml);
    } else {
        tradesListEl.insertAdjacentHTML('beforeend', tradeHtml);
    }

    // Keep only last 10
    while (tradesListEl.children.length > 10) {
        tradesListEl.removeChild(tradesListEl.lastChild);
    }
}

// Add AI reasoning
function addAIReasoning(reasoning, prepend = true) {
    const aiListEl = document.getElementById('ai-reasoning-list');

    const action = reasoning.action || 'hold';
    const aiHtml = `
        <div class="ai-reasoning-item">
            <div class="ai-timestamp">${formatTimestamp(reasoning.timestamp)}</div>
            <div class="ai-decision ${action}">${action.toUpperCase()}: ${reasoning.symbol || 'N/A'}</div>
            <div style="font-size: 13px; color: #ccc;">${reasoning.reasoning || 'No reasoning provided'}</div>
            <div style="font-size: 11px; color: #8892b0; margin-top: 6px;">
                Confidence: ${((reasoning.confidence || 0) * 100).toFixed(0)}% |
                Strategy: ${reasoning.strategy_used || 'N/A'}
            </div>
        </div>
    `;

    if (prepend) {
        aiListEl.insertAdjacentHTML('afterbegin', aiHtml);
    } else {
        aiListEl.insertAdjacentHTML('beforeend', aiHtml);
    }

    // Keep only last 10
    while (aiListEl.children.length > 10) {
        aiListEl.removeChild(aiListEl.lastChild);
    }
}

// Update metrics
function updateMetrics(metrics) {
    // Sharpe ratio
    if (metrics.sharpe_ratio !== undefined) {
        document.getElementById('sharpe-ratio').textContent = metrics.sharpe_ratio.toFixed(2);
    }

    // Win rate
    if (metrics.win_rate !== undefined) {
        document.getElementById('win-rate').textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
    }

    // Total trades
    if (metrics.total_trades !== undefined) {
        document.getElementById('total-trades').textContent = `${metrics.total_trades} trades`;
    }
}

// Initialize portfolio chart
function initPortfolioChart() {
    const ctx = document.getElementById('portfolioChart').getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, CHART_COLORS.gradient1);
    gradient.addColorStop(1, CHART_COLORS.gradient2);

    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: CHART_COLORS.line,
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: CHART_COLORS.line,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: CHART_COLORS.line,
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: (context) => formatCurrency(context.parsed.y)
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: CHART_COLORS.grid,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8892b0',
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: {
                        color: CHART_COLORS.grid,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8892b0',
                        callback: (value) => formatCurrency(value, true)
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

// Update portfolio chart with new data
function updatePortfolioChart(value) {
    const now = new Date().toLocaleTimeString();

    portfolioChart.data.labels.push(now);
    portfolioChart.data.datasets[0].data.push(value);

    // Keep only last 50 data points
    if (portfolioChart.data.labels.length > 50) {
        portfolioChart.data.labels.shift();
        portfolioChart.data.datasets[0].data.shift();
    }

    portfolioChart.update('none'); // Update without animation for performance
}

// Update status badge
function updateStatus(text, isActive) {
    document.getElementById('status-text').textContent = text;
    const statusDot = document.querySelector('.status-dot');
    const statusBadge = document.querySelector('.status-badge');

    if (isActive) {
        statusDot.style.background = '#10b981';
        statusBadge.style.background = 'rgba(16, 185, 129, 0.1)';
        statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else {
        statusDot.style.background = '#ef4444';
        statusBadge.style.background = 'rgba(239, 68, 68, 0.1)';
        statusBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
    }
}

// Utility functions
function formatCurrency(value, compact = false) {
    if (compact && Math.abs(value) >= 1000) {
        return '$' + (value / 1000).toFixed(1) + 'K';
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

function getSignalColor(signal) {
    const colors = {
        'buy': '#10b981',
        'sell': '#ef4444',
        'hold': '#f59e0b',
        'close': '#8892b0'
    };
    return colors[signal.toLowerCase()] || '#8892b0';
}

function getRegimeColor(regime) {
    const colors = {
        'bull': '#10b981',
        'bear': '#ef4444',
        'ranging': '#f59e0b'
    };
    return colors[regime.toLowerCase()] || '#667eea';
}

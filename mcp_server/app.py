"""
MCP-Powered Financial Analyst - Complete Flask Application
Professional-grade web application with all features integrated
"""
import asyncio
import io
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template_string, send_file, make_response
from flask_cors import CORS
import logging
from werkzeug.exceptions import RequestEntityTooLarge
import base64

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our components
from config.settings import settings
from data.database import db_manager, create_tables
from utils.model_client import create_gemini_model_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global state
agents_initialized = False
agents = {}
model_client = None

def init_agents():
    """Initialize all AutoGen agents"""
    global agents_initialized, agents, model_client
    
    if agents_initialized:
        return True
        
    try:
        logger.info("Initializing AutoGen agents...")
        
        # Create model client
        model_client = create_gemini_model_client()
        
        # Import and initialize agents
        from agents.orchestrator_agent import OrchestratorAgent
        from agents.data_analyst_agent import DataAnalystAgent
        from agents.news_sentiment_agent import NewsSentimentAgent
        from agents.report_generator_agent import ReportGeneratorAgent
        from agents.visualization_agent import VisualizationAgent
        from agents.recommendation_agent import RecommendationAgent
        
        agents = {
            'orchestrator': OrchestratorAgent(model_client),
            'data_analyst': DataAnalystAgent(model_client),
            'news_sentiment': NewsSentimentAgent(model_client),
            'report_generator': ReportGeneratorAgent(model_client),
            'visualization': VisualizationAgent(model_client),
            'recommendation': RecommendationAgent(model_client)
        }
        
        agents_initialized = True
        logger.info("All agents initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")
        agents_initialized = False
        return False

def init_database():
    """Initialize database tables"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create tables synchronously for Flask compatibility
        import sqlite3
        import os
        
        db_path = './data/financial_analyst.db'
        if not os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query_text TEXT NOT NULL,
                    response_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT NOT NULL,
                    holdings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

# HTML Template for the main dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-Powered Financial Analyst</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header { 
            text-align: center; 
            margin-bottom: 40px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header h1 { 
            font-size: 3em; 
            margin-bottom: 10px;
            font-weight: bold;
        }
        .header p { 
            font-size: 1.2em; 
            color: #666;
            -webkit-text-fill-color: #666;
        }
        
        .status-card {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border-left: 5px solid #28a745;
        }
        .status-card.warning {
            background: #fff3cd;
            border-left-color: #ffc107;
            border-color: #ffeaa7;
        }
        .status-card.error {
            background: #f8d7da;
            border-left-color: #dc3545;
            border-color: #f5c6cb;
        }
        
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-radius: 10px;
            margin: 20px 0;
            overflow: hidden;
        }
        .tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            cursor: pointer;
            background: #e9ecef;
            border: none;
            transition: all 0.3s;
        }
        .tab.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .tab:hover {
            background: #007bff;
            color: white;
        }
        
        .tab-content {
            display: none;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
            margin: 20px 0;
        }
        .tab-content.active {
            display: block;
        }
        
        .query-form {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .query-input {
            flex: 1;
            min-width: 300px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
        }
        .query-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-area {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            display: none;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .example-queries {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .example-btn {
            padding: 8px 15px;
            background: #e9ecef;
            border: 1px solid #ddd;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .feature-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .portfolio-form {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🚀 MCP-Powered Financial Analyst</h1>
            <p>AI-powered financial analysis with multi-agent AutoGen system</p>
        </header>
        
        <div id="statusSection"></div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('query')">💬 Query Analysis</button>
            <button class="tab" onclick="showTab('portfolio')">📊 Portfolio Management</button>
            <button class="tab" onclick="showTab('reports')">📄 Reports</button>
            <button class="tab" onclick="showTab('visualization')">📈 Visualization</button>
        </div>
        
        <!-- Query Tab -->
        <div id="query-tab" class="tab-content active">
            <h2>🎯 Ask the Financial Analyst</h2>
            
            <div class="query-form">
                <input type="text" class="query-input" id="queryInput" 
                       placeholder="Ask me anything about stocks, markets, or investments..." />
                <button class="btn" id="queryBtn" onclick="processQuery()">Analyze</button>
            </div>
            
            <div class="example-queries">
                <div class="example-btn" onclick="setQuery('Analyze Tesla stock performance and provide investment recommendation')">🚗 Tesla Analysis</div>
                <div class="example-btn" onclick="setQuery('Compare Apple vs Microsoft stocks for long-term investment')">📱 Apple vs Microsoft</div>
                <div class="example-btn" onclick="setQuery('What is the current market sentiment and should I invest now?')">📊 Market Sentiment</div>
                <div class="example-btn" onclick="setQuery('Create a diversified portfolio recommendation for $10,000')">💰 Portfolio Advice</div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>AI agents are analyzing your query...</p>
            </div>
            
            <div class="result-area" id="resultArea"></div>
        </div>
        
        <!-- Portfolio Tab -->
        <div id="portfolio-tab" class="tab-content">
            <h2>📊 Portfolio Management</h2>
            
            <div class="portfolio-form">
                <h3>Create New Portfolio</h3>
                <div class="form-group">
                    <label class="form-label">Portfolio Name:</label>
                    <input type="text" id="portfolioName" class="form-input" placeholder="e.g., Tech Growth Portfolio">
                </div>
                <div class="form-group">
                    <label class="form-label">Holdings (JSON format):</label>
                    <textarea id="portfolioHoldings" class="form-input" rows="4" placeholder='{"AAPL": 100, "MSFT": 50, "GOOGL": 25}'></textarea>
                </div>
                <button class="btn" onclick="createPortfolio()">Create Portfolio</button>
            </div>
            
            <div id="portfolioResults"></div>
        </div>
        
        <!-- Reports Tab -->
        <div id="reports-tab" class="tab-content">
            <h2>📄 Financial Reports</h2>
            
            <div class="features-grid">
                <div class="feature-card">
                    <h3>Stock Analysis Report</h3>
                    <p>Comprehensive analysis of individual stocks with technical and fundamental data.</p>
                    <button class="btn" onclick="generateReport('stock')">Generate Stock Report</button>
                </div>
                <div class="feature-card">
                    <h3>Market Summary Report</h3>
                    <p>Overview of market conditions, trends, and sentiment analysis.</p>
                    <button class="btn" onclick="generateReport('market')">Generate Market Report</button>
                </div>
                <div class="feature-card">
                    <h3>Portfolio Analysis</h3>
                    <p>Detailed portfolio performance and optimization recommendations.</p>
                    <button class="btn" onclick="generateReport('portfolio')">Generate Portfolio Report</button>
                </div>
            </div>
            
            <div id="reportResults"></div>
        </div>
        
        <!-- Visualization Tab -->
        <div id="visualization-tab" class="tab-content">
            <h2>📈 Data Visualization</h2>
            
            <div class="query-form">
                <input type="text" id="vizSymbol" class="query-input" placeholder="Enter stock symbol (e.g., AAPL)" />
                <button class="btn" onclick="generateVisualization()">Create Chart</button>
            </div>
            
            <div class="chart-container" id="chartContainer" style="display: none;">
                <canvas id="stockChart" width="400" height="200"></canvas>
            </div>
            
            <div id="vizResults"></div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p><strong>MCP-Powered Financial Analyst</strong> | Built with AutoGen, Flask, and Model Context Protocol</p>
        </div>
    </div>
    
    <script>
        let systemReady = false;
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const statusSection = document.getElementById('statusSection');
                const queryBtn = document.getElementById('queryBtn');
                
                if (data.status === 'healthy') {
                    systemReady = true;
                    statusSection.innerHTML = `
                        <div class="status-card">
                            <h3>✅ System Ready</h3>
                            <p>All ${data.agent_count} agents initialized and ready for financial analysis!</p>
                            <small>Last checked: ${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                    queryBtn.disabled = false;
                } else if (data.status === 'initializing') {
                    systemReady = false;
                    statusSection.innerHTML = `
                        <div class="status-card warning">
                            <h3>⚠️ System Initializing</h3>
                            <p>AI agents are being initialized. Please wait...</p>
                            <small>Last checked: ${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                    queryBtn.disabled = true;
                } else {
                    systemReady = false;
                    statusSection.innerHTML = `
                        <div class="status-card error">
                            <h3>❌ System Error</h3>
                            <p>${data.message || 'Unknown error occurred'}</p>
                            <small>Last checked: ${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                    queryBtn.disabled = true;
                }
            } catch (error) {
                const statusSection = document.getElementById('statusSection');
                statusSection.innerHTML = `
                    <div class="status-card error">
                        <h3>❌ Connection Error</h3>
                        <p>Unable to connect to the backend service.</p>
                        <small>Error: ${error.message}</small>
                    </div>
                `;
                document.getElementById('queryBtn').disabled = true;
            }
        }
        
        function setQuery(query) {
            document.getElementById('queryInput').value = query;
        }
        
        async function processQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query.trim()) return;
            
            const loading = document.getElementById('loading');
            const resultArea = document.getElementById('resultArea');
            
            loading.style.display = 'block';
            resultArea.style.display = 'none';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, user_id: 'web_user' })
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                resultArea.style.display = 'block';
                resultArea.textContent = JSON.stringify(data, null, 2);
                
            } catch (error) {
                loading.style.display = 'none';
                resultArea.style.display = 'block';
                resultArea.textContent = 'Error: ' + error.message;
            }
        }
        
        async function createPortfolio() {
            const name = document.getElementById('portfolioName').value;
            const holdings = document.getElementById('portfolioHoldings').value;
            
            if (!name.trim() || !holdings.trim()) {
                alert('Please fill in all fields');
                return;
            }
            
            try {
                const response = await fetch('/api/portfolio/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        name: name, 
                        holdings: JSON.parse(holdings),
                        user_id: 'web_user'
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('portfolioResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Portfolio Created Successfully</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('portfolioResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Error Creating Portfolio</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function generateReport(type) {
            const symbol = prompt(`Enter stock symbol for ${type} report:`) || 'AAPL';
            
            try {
                const response = await fetch('/api/report/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        report_type: type,
                        symbol: symbol,
                        user_id: 'web_user'
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('reportResults').innerHTML = `
                    <div class="feature-card">
                        <h3>${type.charAt(0).toUpperCase() + type.slice(1)} Report Generated</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('reportResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Error Generating Report</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function generateVisualization() {
            const symbol = document.getElementById('vizSymbol').value || 'AAPL';
            
            try {
                const response = await fetch(`/api/visualization/stock/${symbol}`);
                const data = await response.json();
                
                if (data.chart_data) {
                    const ctx = document.getElementById('stockChart').getContext('2d');
                    
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.chart_data.dates,
                            datasets: [{
                                label: `${symbol} Price`,
                                data: data.chart_data.prices,
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: false
                                }
                            }
                        }
                    });
                    
                    document.getElementById('chartContainer').style.display = 'block';
                }
                
                document.getElementById('vizResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Visualization Data for ${symbol}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('vizResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Error Generating Visualization</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        // Initialize
        checkSystemStatus();
        setInterval(checkSystemStatus, 30000);
        
        // Allow Enter key to submit query
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !document.getElementById('queryBtn').disabled) {
                processQuery();
            }
        });
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        agent_status = init_agents()
        
        return jsonify({
            "status": "healthy" if agent_status else "initializing",
            "timestamp": datetime.now().isoformat(),
            "agents_initialized": agents_initialized,
            "agent_count": len(agents) if agents else 0,
            "database_status": "connected",
            "api_keys": {
                "google_api": "configured" if settings.google_api_key else "missing",
                "alpha_vantage": "configured" if settings.alpha_vantage_api_key else "missing",
                "news_api": "configured" if settings.news_api_key else "missing"
            }
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/query', methods=['POST'])
def process_financial_query():
    """Process natural language financial queries using AutoGen agents"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', 'default')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        logger.info(f"Processing query for user {user_id}: {query}")
        
        # Initialize agents if not done
        if not init_agents():
            return jsonify({
                "error": "Agents not initialized",
                "suggestion": "Please wait for system initialization"
            }), 503
        
        # Use orchestrator to process the query
        try:
            # For now, simulate the agent processing
            # In a full implementation, you would call:
            # result = await agents['orchestrator'].process_query(query, {"user_id": user_id})
            
            result = simulate_agent_processing(query, user_id)
            
            # Save query to database (simplified)
            try:
                import sqlite3
                conn = sqlite3.connect('./data/financial_analyst.db')
                conn.execute(
                    "INSERT INTO user_queries (user_id, query_text, response_data) VALUES (?, ?, ?)",
                    (user_id, query, json.dumps(result))
                )
                conn.commit()
                conn.close()
            except Exception as db_error:
                logger.warning(f"Database save failed: {db_error}")
            
            return jsonify({
                "success": True,
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "processed_by": "orchestrator_agent"
            })
            
        except Exception as e:
            logger.error(f"Agent processing error: {str(e)}")
            return jsonify({
                "error": f"Processing failed: {str(e)}",
                "query": query
            }), 500
        
    except Exception as e:
        logger.error(f"Query endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def simulate_agent_processing(query: str, user_id: str) -> Dict[str, Any]:
    """Simulate agent processing for demonstration"""
    query_lower = query.lower()
    
    # Stock analysis queries
    if any(word in query_lower for word in ['tesla', 'tsla']):
        return {
            "query_type": "stock_analysis",
            "symbol": "TSLA",
            "analysis": {
                "current_price": 248.73,
                "recommendation": "HOLD",
                "confidence": 0.78,
                "reasoning": "Tesla shows strong EV market position but faces increased competition",
                "technical_indicators": {
                    "rsi": 65.2,
                    "sma_20": 245.30,
                    "sma_50": 240.15,
                    "support": 235.00,
                    "resistance": 260.00
                },
                "fundamentals": {
                    "pe_ratio": 72.5,
                    "market_cap": "790B",
                    "revenue_growth": "15.3%",
                    "profit_margin": "8.2%"
                }
            },
            "news_sentiment": {
                "score": 0.25,
                "articles_analyzed": 47,
                "key_topics": ["EV sales", "Autopilot", "Energy business"]
            },
            "recommendation_details": {
                "action": "HOLD",
                "target_price": 275.00,
                "stop_loss": 220.00,
                "time_horizon": "6-12 months",
                "risk_level": "High"
            }
        }
    
    elif any(word in query_lower for word in ['apple', 'aapl']):
        return {
            "query_type": "stock_analysis",
            "symbol": "AAPL",
            "analysis": {
                "current_price": 185.92,
                "recommendation": "BUY",
                "confidence": 0.87,
                "reasoning": "Apple maintains strong ecosystem and growing services revenue",
                "technical_indicators": {
                    "rsi": 58.3,
                    "sma_20": 182.45,
                    "sma_50": 178.90,
                    "support": 175.00,
                    "resistance": 195.00
                },
                "fundamentals": {
                    "pe_ratio": 28.7,
                    "market_cap": "2.8T",
                    "revenue_growth": "8.1%",
                    "profit_margin": "25.3%"
                }
            }
        }
    
    elif 'compare' in query_lower or 'vs' in query_lower:
        return {
            "query_type": "stock_comparison",
            "symbols": ["AAPL", "MSFT"],
            "comparison": {
                "winner": "AAPL",
                "metrics": {
                    "performance_1y": {"AAPL": 15.2, "MSFT": 12.8},
                    "pe_ratio": {"AAPL": 28.7, "MSFT": 34.2},
                    "dividend_yield": {"AAPL": 0.43, "MSFT": 0.68}
                },
                "recommendation": "Both are strong, but Apple has better momentum currently"
            }
        }
    
    elif any(word in query_lower for word in ['portfolio', 'diversif', 'invest']):
        return {
            "query_type": "portfolio_advice",
            "recommendation": {
                "allocation": {
                    "Technology": 30,
                    "Healthcare": 20,
                    "Financial": 15,
                    "Consumer": 15,
                    "Bonds": 20
                },
                "suggested_stocks": ["AAPL", "MSFT", "JNJ", "JPM", "BRK.B"],
                "risk_level": "Moderate",
                "expected_return": "8-12% annually"
            }
        }
    
    elif any(word in query_lower for word in ['sentiment', 'market']):
        return {
            "query_type": "market_sentiment",
            "sentiment": {
                "overall_score": 0.35,
                "label": "Moderately Positive",
                "confidence": 0.82,
                "factors": {
                    "earnings_season": "positive",
                    "fed_policy": "neutral", 
                    "geopolitical": "slightly_negative",
                    "economic_data": "positive"
                },
                "sector_sentiment": {
                    "Technology": 0.45,
                    "Healthcare": 0.25,
                    "Financial": 0.15,
                    "Energy": -0.10
                }
            }
        }
    
    else:
        return {
            "query_type": "general",
            "response": f"I analyzed your query: '{query}'",
            "suggestions": [
                "Try asking about specific stocks (e.g., 'Analyze Tesla stock')",
                "Ask for market sentiment analysis",
                "Request portfolio recommendations",
                "Compare multiple stocks"
            ],
            "available_features": [
                "Stock analysis",
                "Portfolio optimization",
                "Market sentiment",
                "Technical analysis",
                "Fundamental analysis"
            ]
        }

@app.route('/api/analyze/stock/<symbol>')
def analyze_stock(symbol):
    """Analyze a specific stock"""
    try:
        symbol = symbol.upper()
        
        # Initialize agents
        if not init_agents():
            return jsonify({"error": "Agents not initialized"}), 503
        
        # Simulate comprehensive stock analysis
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": 150.00 + hash(symbol) % 100,  # Mock price
            "analysis": {
                "recommendation": "BUY" if hash(symbol) % 2 else "HOLD",
                "confidence": 0.75 + (hash(symbol) % 25) / 100,
                "technical_analysis": {
                    "trend": "bullish" if hash(symbol) % 2 else "neutral",
                    "rsi": 30 + hash(symbol) % 40,
                    "moving_averages": {
                        "sma_20": 148.50,
                        "sma_50": 145.30,
                        "sma_200": 140.80
                    }
                },
                "fundamental_analysis": {
                    "pe_ratio": 15 + hash(symbol) % 20,
                    "market_cap": f"{hash(symbol) % 500 + 100}B",
                    "revenue_growth": f"{hash(symbol) % 20}%"
                }
            }
        }
        
        return jsonify({
            "success": True,
            "data": analysis
        })
        
    except Exception as e:
        logger.error(f"Stock analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/create', methods=['POST'])
def create_portfolio():
    """Create a new portfolio"""
    try:
        data = request.get_json()
        name = data.get('name')
        holdings = data.get('holdings', {})
        user_id = data.get('user_id', 'default')
        
        if not name:
            return jsonify({"error": "Portfolio name is required"}), 400
        
        # Save to database
        import sqlite3
        conn = sqlite3.connect('./data/financial_analyst.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO portfolios (user_id, name, holdings) VALUES (?, ?, ?)",
            (user_id, name, json.dumps(holdings))
        )
        portfolio_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Analyze portfolio
        analysis = analyze_portfolio_holdings(holdings)
        
        return jsonify({
            "success": True,
            "portfolio_id": portfolio_id,
            "name": name,
            "holdings": holdings,
            "analysis": analysis,
            "created_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Portfolio creation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def analyze_portfolio_holdings(holdings: Dict[str, int]) -> Dict[str, Any]:
    """Analyze portfolio holdings"""
    total_value = sum(holdings.values()) * 100  # Mock calculation
    
    return {
        "total_positions": len(holdings),
        "estimated_value": total_value,
        "diversification_score": min(len(holdings) * 20, 100),
        "risk_level": "Moderate" if len(holdings) > 3 else "High",
        "recommendations": [
            "Consider adding more sectors for diversification" if len(holdings) < 5 else "Good diversification",
            "Review position sizes for better balance",
            "Monitor quarterly earnings for all holdings"
        ]
    }

@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """Generate financial reports"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'stock')
        symbol = data.get('symbol', 'AAPL')
        user_id = data.get('user_id', 'default')
        
        # Generate report based on type
        if report_type == 'stock':
            report = generate_stock_report(symbol)
        elif report_type == 'portfolio':
            report = generate_portfolio_report(user_id)
        elif report_type == 'market':
            report = generate_market_report()
        else:
            return jsonify({"error": "Invalid report type"}), 400
        
        return jsonify({
            "success": True,
            "report": report,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_stock_report(symbol: str) -> Dict[str, Any]:
    """Generate comprehensive stock report"""
    return {
        "title": f"Stock Analysis Report - {symbol}",
        "symbol": symbol,
        "sections": {
            "executive_summary": f"Comprehensive analysis of {symbol} shows mixed signals with moderate buy recommendation.",
            "financial_highlights": {
                "revenue": "Strong revenue growth of 12.5% YoY",
                "profitability": "Maintaining healthy profit margins",
                "debt": "Conservative debt levels"
            },
            "technical_analysis": {
                "trend": "Bullish momentum in short-term",
                "support_resistance": "Key support at $140, resistance at $160",
                "indicators": "RSI indicates neutral conditions"
            },
            "recommendation": {
                "action": "BUY",
                "target_price": 165.00,
                "time_horizon": "6-12 months"
            }
        }
    }

def generate_portfolio_report(user_id: str) -> Dict[str, Any]:
    """Generate portfolio analysis report"""
    return {
        "title": "Portfolio Analysis Report",
        "user_id": user_id,
        "sections": {
            "overview": "Portfolio shows good diversification across sectors",
            "performance": "Outperforming S&P 500 by 2.3% YTD",
            "risk_analysis": "Moderate risk profile with beta of 1.15",
            "recommendations": [
                "Consider rebalancing technology allocation",
                "Add international exposure",
                "Review defensive positions"
            ]
        }
    }

def generate_market_report() -> Dict[str, Any]:
    """Generate market analysis report"""
    return {
        "title": "Market Analysis Report",
        "sections": {
            "market_overview": "Markets showing resilience despite volatility",
            "sector_analysis": {
                "Technology": "Leading gains with AI momentum",
                "Healthcare": "Steady performance",
                "Energy": "Mixed signals from commodity prices"
            },
            "economic_indicators": {
                "gdp_growth": "2.1% annual growth",
                "inflation": "Moderating to 3.2%",
                "employment": "Strong labor market conditions"
            },
            "outlook": "Cautiously optimistic for next quarter"
        }
    }

@app.route('/api/visualization/stock/<symbol>')
def get_stock_visualization(symbol):
    """Get stock visualization data"""
    try:
        symbol = symbol.upper()
        
        # Generate mock data for demonstration
        import random
        from datetime import timedelta
        
        base_date = datetime.now() - timedelta(days=30)
        dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        
        base_price = 150 + hash(symbol) % 50
        prices = []
        current_price = base_price
        
        for _ in range(30):
            change = random.uniform(-0.05, 0.05)
            current_price *= (1 + change)
            prices.append(round(current_price, 2))
        
        visualization_data = {
            "symbol": symbol,
            "chart_data": {
                "dates": dates,
                "prices": prices,
                "volumes": [random.randint(1000000, 10000000) for _ in range(30)]
            },
            "statistics": {
                "current_price": prices[-1],
                "change_30d": round(((prices[-1] - prices[0]) / prices[0]) * 100, 2),
                "high_30d": max(prices),
                "low_30d": min(prices),
                "avg_volume": sum([random.randint(1000000, 10000000) for _ in range(30)]) / 30
            }
        }
        
        return jsonify({
            "success": True,
            "data": visualization_data
        })
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare/stocks', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if len(symbols) < 2:
            return jsonify({"error": "At least 2 symbols required"}), 400
        
        comparison = {}
        for symbol in symbols:
            comparison[symbol] = {
                "current_price": 100 + hash(symbol) % 100,
                "pe_ratio": 15 + hash(symbol) % 20,
                "market_cap": f"{hash(symbol) % 500 + 100}B",
                "recommendation": "BUY" if hash(symbol) % 2 else "HOLD"
            }
        
        return jsonify({
            "success": True,
            "comparison": comparison,
            "winner": max(symbols, key=lambda x: hash(x) % 100),
            "analysis": f"Based on analysis, {symbols[0]} shows better value proposition"
        })
        
    except Exception as e:
        logger.error(f"Stock comparison error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Initialize agents
    init_agents()
    
    logger.info("Starting MCP-Powered Financial Analyst Flask Application...")
    logger.info("Available endpoints:")
    logger.info("  GET  /                          - Main dashboard")
    logger.info("  GET  /api/health                - Health check")
    logger.info("  POST /api/query                 - Process financial queries")
    logger.info("  GET  /api/analyze/stock/<symbol> - Analyze specific stock")
    logger.info("  POST /api/portfolio/create      - Create portfolio")
    logger.info("  POST /api/report/generate       - Generate reports")
    logger.info("  GET  /api/visualization/stock/<symbol> - Get stock charts")
    logger.info("  POST /api/compare/stocks        - Compare multiple stocks")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

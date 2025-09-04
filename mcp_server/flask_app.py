"""
Flask Application for MCP-Powered Financial Analyst
Simple and reliable web interface
"""
import asyncio
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
import os

# Import our agents and configuration
from config.settings import settings
from data.database import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global agents - will be initialized on startup
agents = {}

def init_agents():
    """Initialize all AutoGen agents"""
    global agents
    
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        
        # Initialize model client
        model_client = OpenAIChatCompletionClient(
            model="gemini-2.0-flash-exp",
            api_key=settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Import and initialize agents
        from agents.orchestrator_agent import OrchestratorAgent
        from agents.data_analyst_agent import DataAnalystAgent
        from agents.news_sentiment_agent import NewsSentimentAgent
        from agents.report_generator_agent import ReportGeneratorAgent
        from agents.visualization_agent import VisualizationAgent
        from agents.recommendation_agent import RecommendationAgent
        
        agents['orchestrator'] = OrchestratorAgent(model_client)
        agents['data_analyst'] = DataAnalystAgent(model_client)
        agents['news_sentiment'] = NewsSentimentAgent(model_client)
        agents['report_generator'] = ReportGeneratorAgent(model_client)
        agents['visualization'] = VisualizationAgent(model_client)
        agents['recommendation'] = RecommendationAgent(model_client)
        
        logger.info("All agents initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")
        return False

# HTML template for the main page
MAIN_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-Powered Financial Analyst</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
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
        
        .features { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 25px; 
            margin: 40px 0;
        }
        .feature-card { 
            background: white;
            padding: 25px; 
            border-radius: 15px; 
            border-left: 5px solid #667eea;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .feature-card h3 { 
            color: #333; 
            margin-bottom: 15px; 
            font-size: 1.3em;
        }
        .feature-card p { 
            color: #666; 
            line-height: 1.6;
        }
        
        .demo-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }
        .demo-section h2 {
            color: #333;
            margin-bottom: 20px;
            text-align: center;
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
        .query-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .query-btn:hover {
            transform: scale(1.05);
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
        
        .result-area {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            display: none;
        }
        
        .api-endpoints {
            background: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }
        .api-endpoints h2 {
            margin-bottom: 20px;
            text-align: center;
        }
        .endpoint {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .method {
            background: #3498db;
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .method.post { background: #e74c3c; }
        .method.get { background: #27ae60; }
        
        .status-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #27ae60;
            color: white;
            border-radius: 25px;
            font-size: 14px;
        }
        .status-indicator.offline {
            background: #e74c3c;
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
    </style>
</head>
<body>
    <div class="status-indicator" id="statusIndicator">🟢 System Online</div>
    
    <div class="container">
        <header class="header">
            <h1>🚀 MCP-Powered Financial Analyst</h1>
            <p>AI-powered financial analysis with multi-agent AutoGen system</p>
        </header>
        
        <div class="demo-section">
            <h2>🎯 Try It Now - Ask Any Financial Question</h2>
            
            <div class="query-form">
                <input type="text" class="query-input" id="queryInput" 
                       placeholder="Ask me anything about stocks, markets, or investments..." />
                <button class="query-btn" onclick="processQuery()">Analyze</button>
            </div>
            
            <div class="example-queries">
                <div class="example-btn" onclick="setQuery('Analyze Apple stock performance')">📊 Analyze Apple</div>
                <div class="example-btn" onclick="setQuery('Compare Tesla and Ford stocks')">⚖️ Compare Stocks</div>
                <div class="example-btn" onclick="setQuery('What is the market sentiment today?')">📰 Market Sentiment</div>
                <div class="example-btn" onclick="setQuery('Should I buy Amazon stock?')">🎯 Investment Advice</div>
                <div class="example-btn" onclick="setQuery('Create a portfolio analysis report')">📄 Generate Report</div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing your query with AI agents...</p>
            </div>
            
            <div class="result-area" id="resultArea"></div>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h3>📊 Real-time Stock Analysis</h3>
                <p>Get comprehensive analysis of any stock with technical indicators, financial metrics, and AI-powered insights using real-time data from Alpha Vantage.</p>
            </div>
            <div class="feature-card">
                <h3>📰 Sentiment Analysis</h3>
                <p>Monitor market sentiment from news sources and social media. Our AI agents analyze thousands of articles to gauge market mood and investor confidence.</p>
            </div>
            <div class="feature-card">
                <h3>🎯 Investment Recommendations</h3>
                <p>Receive personalized buy/sell/hold recommendations based on multi-factor analysis including technical indicators, fundamentals, and market sentiment.</p>
            </div>
            <div class="feature-card">
                <h3>📈 Portfolio Optimization</h3>
                <p>Optimize your investment portfolio using Modern Portfolio Theory. Get rebalancing recommendations and risk assessments for better returns.</p>
            </div>
            <div class="feature-card">
                <h3>📄 Professional Reports</h3>
                <p>Generate comprehensive financial reports in PDF or HTML format with charts, analysis, and actionable recommendations for your investments.</p>
            </div>
            <div class="feature-card">
                <h3>🤖 Multi-Agent AI System</h3>
                <p>Powered by 6 specialized AutoGen agents working together: Orchestrator, Data Analyst, Sentiment Analyzer, Report Generator, Visualizer, and Recommender.</p>
            </div>
        </div>
        
        <div class="api-endpoints">
            <h2>🔗 API Endpoints</h2>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/query</strong>
                <p>Process natural language financial queries</p>
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/analyze/stock</strong>
                <p>Comprehensive stock analysis with technical and fundamental data</p>
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/analyze/portfolio</strong>
                <p>Portfolio analysis and optimization recommendations</p>
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/compare/stocks</strong>
                <p>Side-by-side comparison of multiple stocks</p>
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/sentiment/market</strong>
                <p>Current market sentiment analysis from news and social media</p>
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/health</strong>
                <p>System health check and agent status</p>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p>Built with AutoGen, Flask, and Model Context Protocol (MCP)</p>
            <p>© 2025 MCP-Powered Financial Analyst</p>
        </div>
    </div>
    
    <script>
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
        
        // Check system status
        async function checkStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                const indicator = document.getElementById('statusIndicator');
                
                if (data.status === 'healthy') {
                    indicator.textContent = '🟢 System Online';
                    indicator.className = 'status-indicator';
                } else {
                    indicator.textContent = '🔴 System Offline';
                    indicator.className = 'status-indicator offline';
                }
            } catch (error) {
                const indicator = document.getElementById('statusIndicator');
                indicator.textContent = '🔴 System Offline';
                indicator.className = 'status-indicator offline';
            }
        }
        
        // Check status on page load and every 30 seconds
        checkStatus();
        setInterval(checkStatus, 30000);
        
        // Allow Enter key to submit query
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                processQuery();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the main dashboard"""
    return render_template_string(MAIN_PAGE_HTML)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    agent_status = {}
    for name, agent in agents.items():
        agent_status[name] = "online" if agent else "offline"
    
    return jsonify({
        "status": "healthy" if agents else "initializing",
        "timestamp": datetime.now().isoformat(),
        "agents": agent_status,
        "apis": {
            "alpha_vantage": "configured" if settings.alpha_vantage_api_key != "your_alpha_vantage_key_here" else "not_configured",
            "news_api": "configured" if settings.news_api_key != "your_news_api_key_here" else "not_configured",
            "twitter_api": "configured" if settings.twitter_bearer_token not in ["your_actual_bearer_token_here", "DISABLED_FOR_NOW"] else "not_configured"
        }
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language financial queries"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', 'default')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if not agents.get('orchestrator'):
            return jsonify({"error": "Orchestrator agent not available"}), 500
        
        logger.info(f"Processing query: {query}")
        
        # For now, return a mock response since async processing in Flask is complex
        # In a real implementation, you'd use Celery or similar for async processing
        
        result = {
            "query_type": "stock_analysis" if any(word in query.lower() for word in ['stock', 'analyze', 'price']) else "general",
            "symbols": ["AAPL"] if "apple" in query.lower() else ["TSLA"] if "tesla" in query.lower() else [],
            "analysis": "Query processed successfully",
            "recommendations": ["This is a mock response for demonstration"],
            "confidence": 0.85
        }
        
        return jsonify({
            "success": True,
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/stock', methods=['POST'])
def analyze_stock():
    """Analyze a single stock"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        analysis_type = data.get('analysis_type', 'comprehensive')
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        logger.info(f"Analyzing stock: {symbol}")
        
        # Mock response for demonstration
        mock_analysis = {
            "symbol": symbol,
            "current_price": 150.25 if symbol == "AAPL" else 250.75,
            "daily_change": 2.5 if symbol == "AAPL" else -1.2,
            "key_metrics": {
                "pe_ratio": 25.4,
                "market_cap": 2400000000000,
                "52w_high": 180.0,
                "52w_low": 120.0,
                "volume": 50000000
            },
            "technical_analysis": {
                "rsi": 65.2,
                "sma_20": 145.0,
                "sma_50": 140.0,
                "trend": "bullish"
            },
            "recommendation": "BUY",
            "confidence": 0.82
        }
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "analysis": mock_analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing stock: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment/market')
def market_sentiment():
    """Get market sentiment"""
    try:
        # Mock sentiment data
        mock_sentiment = {
            "overall_sentiment": {
                "score": 0.15,
                "label": "slightly positive",
                "confidence": 0.78
            },
            "news_articles_analyzed": 150,
            "social_media_mentions": 2500,
            "trending_topics": ["AI stocks", "Federal Reserve", "Tech earnings"],
            "sector_sentiment": {
                "technology": 0.25,
                "healthcare": 0.10,
                "financial": -0.05,
                "energy": -0.15
            }
        }
        
        return jsonify({
            "success": True,
            "market_sentiment": mock_sentiment,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare/stocks', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols or len(symbols) < 2:
            return jsonify({"error": "At least 2 symbols required for comparison"}), 400
        
        logger.info(f"Comparing stocks: {symbols}")
        
        # Mock comparison data
        mock_comparison = {
            "symbols": symbols,
            "comparison_metrics": {
                "performance_1y": {symbols[0]: 15.2, symbols[1]: -5.8},
                "pe_ratio": {symbols[0]: 25.4, symbols[1]: 18.7},
                "market_cap": {symbols[0]: 2400000000000, symbols[1]: 800000000000}
            },
            "winner": symbols[0],
            "recommendation": f"Based on analysis, {symbols[0]} shows better fundamentals and momentum"
        }
        
        return jsonify({
            "success": True,
            "comparison": mock_comparison,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    try:
        # Create database tables
        from data.database import create_tables
        create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
    
    # Initialize agents
    if init_agents():
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to initialize agents. Starting with limited functionality...")
        app.run(host='0.0.0.0', port=5000, debug=True)

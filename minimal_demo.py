"""
Minimal Financial Analyst Demo - No External Dependencies
Uses only Python standard library for maximum compatibility
"""
import json
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalystHandler(http.server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_dashboard()
        elif parsed_path.path == '/api/health':
            self.serve_health_check()
        elif parsed_path.path == '/api/demo/stocks':
            self.serve_demo_stocks()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/query':
            self.handle_query()
        else:
            self.send_error(404, "Not Found")
    
    def serve_dashboard(self):
        """Serve the main dashboard"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-Powered Financial Analyst - Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1000px; 
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
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
        
        .stock-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stock-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stock-card.negative {
            border-left-color: #dc3545;
        }
        .stock-symbol {
            font-weight: bold;
            font-size: 1.2em;
            color: #333;
        }
        .stock-price {
            font-size: 1.4em;
            margin: 5px 0;
        }
        .stock-change {
            font-size: 0.9em;
        }
        .stock-change.positive { color: #28a745; }
        .stock-change.negative { color: #dc3545; }
        
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
    <div class="container">
        <header class="header">
            <h1>🚀 MCP-Powered Financial Analyst</h1>
            <p>AI-powered financial analysis system - Demo Version</p>
        </header>
        
        <div class="status-card">
            <h3>✅ Demo Server Running</h3>
            <p>This is a demonstration version running with Python standard library only.</p>
            <p><strong>Status:</strong> Ready for queries | <strong>Mode:</strong> Demo | <strong>Time:</strong> <span id="currentTime"></span></p>
        </div>
        
        <div class="demo-section">
            <h2>📊 Market Overview</h2>
            <div class="stock-grid" id="stockGrid">
                <!-- Stocks will be loaded here -->
            </div>
        </div>
        
        <div class="demo-section">
            <h2>🎯 Ask the Financial Analyst</h2>
            
            <div class="query-form">
                <input type="text" class="query-input" id="queryInput" 
                       placeholder="Ask me about stocks, markets, or investments..." />
                <button class="query-btn" onclick="processQuery()">Analyze</button>
            </div>
            
            <div class="example-queries">
                <div class="example-btn" onclick="setQuery('Analyze Apple stock')">📊 Apple Analysis</div>
                <div class="example-btn" onclick="setQuery('Should I invest in Tesla?')">⚡ Tesla Advice</div>
                <div class="example-btn" onclick="setQuery('Market sentiment today')">📰 Market Sentiment</div>
                <div class="example-btn" onclick="setQuery('Best tech stocks')">💻 Tech Stocks</div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing your query...</p>
            </div>
            
            <div class="result-area" id="resultArea"></div>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h3>📊 Real-time Analysis</h3>
                <p>Get comprehensive stock analysis with AI-powered insights and recommendations.</p>
            </div>
            <div class="feature-card">
                <h3>🤖 Multi-Agent System</h3>
                <p>Powered by specialized AutoGen agents for comprehensive financial analysis.</p>
            </div>
            <div class="feature-card">
                <h3>📈 Market Intelligence</h3>
                <p>Monitor market trends, sentiment, and receive personalized investment advice.</p>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p><strong>MCP-Powered Financial Analyst Demo</strong></p>
            <p>Built with Python • AutoGen • Model Context Protocol</p>
        </div>
    </div>
    
    <script>
        function updateTime() {
            document.getElementById('currentTime').textContent = new Date().toLocaleTimeString();
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
                    body: JSON.stringify({ query: query, user_id: 'demo_user' })
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
        
        async function loadStocks() {
            try {
                const response = await fetch('/api/demo/stocks');
                const data = await response.json();
                
                const stockGrid = document.getElementById('stockGrid');
                stockGrid.innerHTML = '';
                
                data.trending_stocks.forEach(stock => {
                    const isPositive = stock.change.startsWith('+');
                    const stockCard = document.createElement('div');
                    stockCard.className = 'stock-card' + (isPositive ? '' : ' negative');
                    stockCard.innerHTML = `
                        <div class="stock-symbol">${stock.symbol}</div>
                        <div class="stock-price">$${stock.price}</div>
                        <div class="stock-change ${isPositive ? 'positive' : 'negative'}">${stock.change}</div>
                    `;
                    stockGrid.appendChild(stockCard);
                });
                
            } catch (error) {
                console.error('Error loading stocks:', error);
            }
        }
        
        // Initialize
        updateTime();
        setInterval(updateTime, 1000);
        loadStocks();
        
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
        self.wfile.write(html.encode())
    
    def serve_health_check(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        health_data = {
            "status": "healthy",
            "server": "minimal_demo",
            "message": "Demo server running with standard library",
            "timestamp": datetime.now().isoformat(),
            "features": ["stock_analysis", "market_data", "demo_queries"]
        }
        self.wfile.write(json.dumps(health_data, indent=2).encode())
    
    def serve_demo_stocks(self):
        """Demo stock data"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        demo_data = {
            "trending_stocks": [
                {"symbol": "AAPL", "price": "185.92", "change": "+2.34%"},
                {"symbol": "TSLA", "price": "248.73", "change": "-1.82%"},
                {"symbol": "MSFT", "price": "441.06", "change": "+0.97%"},
                {"symbol": "GOOGL", "price": "188.54", "change": "+1.15%"},
                {"symbol": "AMZN", "price": "197.34", "change": "+0.58%"},
                {"symbol": "NVDA", "price": "892.45", "change": "+3.21%"}
            ],
            "market_summary": {
                "sp500": {"value": "5954.05", "change": "+0.38%"},
                "nasdaq": {"value": "19161.63", "change": "+0.77%"},
                "dow": {"value": "42840.26", "change": "+0.25%"}
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.wfile.write(json.dumps(demo_data, indent=2).encode())
    
    def handle_query(self):
        """Handle financial queries"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            query_data = json.loads(post_data.decode())
            
            query = query_data.get('query', '').lower()
            user_id = query_data.get('user_id', 'demo')
            
            # Simple query analysis
            analysis_result = self.analyze_query(query)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "success": True,
                "query": query,
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat(),
                "server": "minimal_demo"
            }
            
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            self.send_error(500, f"Query processing error: {str(e)}")
    
    def analyze_query(self, query):
        """Simple query analysis logic"""
        query = query.lower()
        
        # Stock-specific analysis
        if 'apple' in query or 'aapl' in query:
            return {
                "stock": "AAPL",
                "current_price": 185.92,
                "recommendation": "BUY",
                "confidence": 0.85,
                "reasoning": "Apple shows strong fundamentals with robust ecosystem and growing services revenue",
                "key_metrics": {
                    "pe_ratio": 28.7,
                    "market_cap": "2.8T",
                    "52w_change": "+12.4%"
                }
            }
        elif 'tesla' in query or 'tsla' in query:
            return {
                "stock": "TSLA",
                "current_price": 248.73,
                "recommendation": "HOLD",
                "confidence": 0.72,
                "reasoning": "Tesla shows innovation leadership but faces increased competition in EV market",
                "key_metrics": {
                    "pe_ratio": 72.5,
                    "market_cap": "790B",
                    "52w_change": "-8.2%"
                }
            }
        elif 'microsoft' in query or 'msft' in query:
            return {
                "stock": "MSFT",
                "current_price": 441.06,
                "recommendation": "BUY",
                "confidence": 0.88,
                "reasoning": "Microsoft continues strong performance in cloud computing and AI integration",
                "key_metrics": {
                    "pe_ratio": 34.2,
                    "market_cap": "3.3T",
                    "52w_change": "+15.8%"
                }
            }
        elif 'sentiment' in query or 'market' in query:
            return {
                "market_sentiment": "Moderately Positive",
                "sentiment_score": 0.65,
                "key_factors": [
                    "Strong corporate earnings",
                    "Fed policy uncertainty",
                    "Geopolitical tensions",
                    "AI sector growth"
                ],
                "recommendation": "Cautiously optimistic - maintain diversified portfolio"
            }
        elif 'tech' in query or 'technology' in query:
            return {
                "sector": "Technology",
                "outlook": "Positive",
                "top_picks": ["AAPL", "MSFT", "GOOGL", "NVDA"],
                "reasoning": "AI revolution driving growth, cloud adoption accelerating",
                "risk_factors": ["Regulatory scrutiny", "Valuation concerns"]
            }
        else:
            return {
                "query_type": "general",
                "response": f"I analyzed your query: '{query}'",
                "suggestion": "Try asking about specific stocks (Apple, Tesla, Microsoft) or market sentiment",
                "available_features": [
                    "Individual stock analysis",
                    "Market sentiment analysis", 
                    "Sector recommendations",
                    "Investment advice"
                ]
            }

if __name__ == '__main__':
    PORT = 8080
    
    print("=" * 60)
    print("🚀 MCP-Powered Financial Analyst - Demo Server")
    print("=" * 60)
    print(f"Starting server on http://localhost:{PORT}")
    print("Features:")
    print("  ✅ Stock Analysis (AAPL, TSLA, MSFT)")
    print("  ✅ Market Sentiment Analysis")
    print("  ✅ Investment Recommendations")
    print("  ✅ Real-time Demo Data")
    print("  ✅ Interactive Web Interface")
    print("-" * 60)
    print("This is a demonstration version using Python standard library.")
    print("Full version with AutoGen agents will be available after package installation.")
    print("-" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        with socketserver.TCPServer(("", PORT), FinancialAnalystHandler) as httpd:
            print(f"✅ Server running at http://localhost:{PORT}/")
            print("🌐 Open your browser and navigate to the URL above")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

"""
Main Application Entry Point for MCP-Powered Financial Analyst
"""
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
import logging
from datetime import datetime

# Import our agents
from agents.orchestrator_agent import OrchestratorAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.visualization_agent import VisualizationAgent
from agents.recommendation_agent import RecommendationAgent

# Import configuration and database
from config.settings import settings
from data.database import db_manager
from utils.model_client import create_gemini_model_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MCP-Powered Financial Analyst",
    description="AI-powered financial analysis with multi-agent AutoGen system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class StockAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"

class PortfolioAnalysisRequest(BaseModel):
    portfolio: Dict[str, Dict[str, float]]  # {symbol: {shares: float, avg_cost: float}}

class ComparisonRequest(BaseModel):
    symbols: List[str]

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"

class ReportRequest(BaseModel):
    analysis_type: str
    data: Dict[str, Any]
    format: str = "html"

# Global agents
orchestrator_agent = None
data_analyst_agent = None
news_sentiment_agent = None
report_generator_agent = None
visualization_agent = None
recommendation_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global orchestrator_agent, data_analyst_agent, news_sentiment_agent
    global report_generator_agent, visualization_agent, recommendation_agent
    
    logger.info("Starting MCP-Powered Financial Analyst...")
    
    try:
        # Initialize database
        await db_manager.init_db()
        logger.info("Database initialized successfully")
        
        # Initialize model client for AutoGen
        model_client = create_gemini_model_client()
        
        # Initialize all agents
        orchestrator_agent = OrchestratorAgent(model_client)
        data_analyst_agent = DataAnalystAgent(model_client)
        news_sentiment_agent = NewsSentimentAgent(model_client)
        report_generator_agent = ReportGeneratorAgent(model_client)
        visualization_agent = VisualizationAgent(model_client)
        recommendation_agent = RecommendationAgent(model_client)
        
        logger.info("All agents initialized successfully")
        
        # Create output directories
        os.makedirs(settings.reports_output_dir, exist_ok=True)
        os.makedirs(settings.charts_output_dir, exist_ok=True)
        
        logger.info("MCP-Powered Financial Analyst started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down MCP-Powered Financial Analyst...")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP-Powered Financial Analyst</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .feature-card { background: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }
            .feature-card h3 { color: #2c3e50; margin-top: 0; }
            .api-section { margin-top: 30px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }
            .method { background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 MCP-Powered Financial Analyst</h1>
                <p>AI-powered financial analysis with multi-agent AutoGen system</p>
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>📊 Stock Analysis</h3>
                    <p>Comprehensive technical and fundamental analysis with real-time data from Alpha Vantage and Yahoo Finance.</p>
                </div>
                <div class="feature-card">
                    <h3>📰 Sentiment Analysis</h3>
                    <p>Market sentiment analysis from financial news sources using advanced NLP techniques.</p>
                </div>
                <div class="feature-card">
                    <h3>📈 Portfolio Optimization</h3>
                    <p>Modern Portfolio Theory-based optimization with risk assessment and rebalancing recommendations.</p>
                </div>
                <div class="feature-card">
                    <h3>📄 Automated Reports</h3>
                    <p>Professional PDF and HTML reports with charts, analysis, and recommendations.</p>
                </div>
                <div class="feature-card">
                    <h3>🎯 Investment Recommendations</h3>
                    <p>AI-powered buy/sell/hold recommendations with confidence scores and reasoning.</p>
                </div>
                <div class="feature-card">
                    <h3>📊 Interactive Visualizations</h3>
                    <p>Dynamic charts and graphs for technical analysis, portfolio allocation, and performance tracking.</p>
                </div>
            </div>
            
            <div class="api-section">
                <h2>🔗 API Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/analyze/stock</strong>
                    <p>Analyze a single stock with comprehensive metrics</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/analyze/portfolio</strong>
                    <p>Analyze portfolio performance and optimization</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/compare/stocks</strong>
                    <p>Compare multiple stocks side by side</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/query</strong>
                    <p>Natural language financial queries (e.g., "Compare Tesla and Ford earnings")</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/generate/report</strong>
                    <p>Generate professional financial reports</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/health</strong>
                    <p>Check system health and agent status</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/docs</strong>
                    <p>Interactive API documentation (Swagger UI)</p>
                </div>
            </div>
            
            <div style="margin-top: 30px; text-align: center; color: #7f8c8d;">
                <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
                <p>Built with AutoGen, FastAPI, and Model Context Protocol (MCP)</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "orchestrator": "online" if orchestrator_agent else "offline",
            "data_analyst": "online" if data_analyst_agent else "offline",
            "news_sentiment": "online" if news_sentiment_agent else "offline",
            "report_generator": "online" if report_generator_agent else "offline",
            "visualization": "online" if visualization_agent else "offline",
            "recommendation": "online" if recommendation_agent else "offline"
        },
        "apis": {
            "alpha_vantage": "configured" if settings.alpha_vantage_api_key != "your_alpha_vantage_key_here" else "not_configured",
            "news_api": "configured" if settings.news_api_key != "your_news_api_key_here" else "not_configured",
            "twitter_api": "configured" if settings.twitter_bearer_token != "your_actual_bearer_token_here" else "not_configured"
        }
    }

@app.post("/api/query")
async def process_natural_language_query(request: QueryRequest):
    """Process natural language financial queries"""
    try:
        if not orchestrator_agent:
            raise HTTPException(status_code=500, detail="Orchestrator agent not initialized")
        
        logger.info(f"Processing query: {request.query}")
        
        # Process query through orchestrator
        result = await orchestrator_agent.process_query(request.query, request.user_id)
        
        # Save query to database
        await db_manager.save_user_query(request.user_id, request.query, result)
        
        return {
            "success": True,
            "query": request.query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a single stock"""
    try:
        if not data_analyst_agent:
            raise HTTPException(status_code=500, detail="Data analyst agent not initialized")
        
        logger.info(f"Analyzing stock: {request.symbol}")
        
        # Perform stock analysis
        analysis_result = await data_analyst_agent.analyze_stock(
            request.symbol, 
            request.analysis_type
        )
        
        # Get sentiment analysis
        sentiment_result = await news_sentiment_agent.analyze_stock_sentiment(request.symbol)
        
        # Generate recommendation
        recommendation_result = await recommendation_agent.analyze_stock_recommendation(analysis_result)
        
        return {
            "success": True,
            "symbol": request.symbol,
            "analysis": analysis_result,
            "sentiment": sentiment_result,
            "recommendation": recommendation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing stock {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/portfolio")
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """Analyze portfolio performance and optimization"""
    try:
        if not data_analyst_agent or not recommendation_agent:
            raise HTTPException(status_code=500, detail="Required agents not initialized")
        
        logger.info(f"Analyzing portfolio with {len(request.portfolio)} positions")
        
        # Perform portfolio analysis
        portfolio_analysis = await data_analyst_agent.get_portfolio_analysis(request.portfolio)
        
        # Generate optimization recommendations
        optimization_result = await recommendation_agent.optimize_portfolio(portfolio_analysis)
        
        # Generate diversification recommendations
        diversification_result = await recommendation_agent.generate_diversification_recommendations(portfolio_analysis)
        
        return {
            "success": True,
            "portfolio_analysis": portfolio_analysis,
            "optimization": optimization_result,
            "diversification": diversification_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare/stocks")
async def compare_stocks(request: ComparisonRequest):
    """Compare multiple stocks"""
    try:
        if not data_analyst_agent or not recommendation_agent:
            raise HTTPException(status_code=500, detail="Required agents not initialized")
        
        logger.info(f"Comparing stocks: {request.symbols}")
        
        # Perform stock comparison
        comparison_result = await data_analyst_agent.compare_stocks(request.symbols)
        
        # Generate investment recommendations
        investment_comparison = await recommendation_agent.compare_investment_options(comparison_result)
        
        # Compare sentiments
        sentiment_comparison = await news_sentiment_agent.compare_stock_sentiments(request.symbols)
        
        return {
            "success": True,
            "symbols": request.symbols,
            "comparison": comparison_result,
            "investment_analysis": investment_comparison,
            "sentiment_comparison": sentiment_comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/report")
async def generate_report(request: ReportRequest):
    """Generate financial reports"""
    try:
        if not report_generator_agent:
            raise HTTPException(status_code=500, detail="Report generator agent not initialized")
        
        logger.info(f"Generating {request.analysis_type} report in {request.format} format")
        
        # Generate report based on type
        if request.analysis_type == "stock_analysis":
            report_result = await report_generator_agent.generate_stock_analysis_report(
                request.data, request.format
            )
        elif request.analysis_type == "portfolio_analysis":
            report_result = await report_generator_agent.generate_portfolio_report(
                request.data, request.format
            )
        elif request.analysis_type == "sentiment_analysis":
            report_result = await report_generator_agent.generate_market_sentiment_report(
                request.data, request.format
            )
        elif request.analysis_type == "comparison":
            report_result = await report_generator_agent.generate_comparison_report(
                request.data, request.format
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")
        
        return {
            "success": True,
            "report": report_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/chart")
async def generate_chart(symbol: str, chart_type: str = "candlestick"):
    """Generate stock chart visualizations"""
    try:
        if not visualization_agent or not data_analyst_agent:
            raise HTTPException(status_code=500, detail="Required agents not initialized")
        
        logger.info(f"Generating {chart_type} chart for {symbol}")
        
        # Get historical data
        analysis_result = await data_analyst_agent.analyze_stock(symbol)
        
        # Generate chart
        chart_result = await visualization_agent.create_stock_price_chart(
            analysis_result.get("historical_data", {}), 
            symbol, 
            chart_type
        )
        
        return {
            "success": True,
            "chart": chart_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/market")
async def get_market_sentiment():
    """Get current market sentiment analysis"""
    try:
        if not news_sentiment_agent:
            raise HTTPException(status_code=500, detail="News sentiment agent not initialized")
        
        logger.info("Analyzing market sentiment")
        
        # Get market sentiment
        sentiment_result = await news_sentiment_agent.analyze_market_sentiment()
        
        return {
            "success": True,
            "market_sentiment": sentiment_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (charts, reports)
app.mount("/static", StaticFiles(directory="reports"), name="static")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

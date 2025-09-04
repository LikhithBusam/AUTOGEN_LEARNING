"""
Main Application Entry Point for MCP-Powered Financial Analyst
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import configuration and database
from config.settings import settings
from data.database import db_manager
from utils.model_client import create_gemini_model_client

# Import our agents
from agents.orchestrator_agent import OrchestratorAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.visualization_agent import VisualizationAgent
from agents.recommendation_agent import RecommendationAgent

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.log_file, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("mcp_fin_analyst")

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="MCP-Powered Financial Analyst",
    description="AI-powered financial analysis with a multi-agent AutoGen system",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------------------
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    analysis_type: str = Field("comprehensive", description="Type of analysis")

class PortfolioAnalysisRequest(BaseModel):
    # Example: {"AAPL": {"shares": 10, "avg_cost": 180.5}, "MSFT": {"shares": 5, "avg_cost": 305.0}}
    portfolio: Dict[str, Dict[str, float]]

class ComparisonRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=2)

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"

class ReportRequest(BaseModel):
    analysis_type: str
    data: Dict[str, Any]
    format: str = Field("html", description="html or pdf")

class ChartRequest(BaseModel):
    symbol: str
    chart_type: str = Field("candlestick", description="candlestick | line | ohlc")

# ------------------------------------------------------------------------------
# Globals (agents)
# ------------------------------------------------------------------------------
orchestrator_agent: Optional[OrchestratorAgent] = None
data_analyst_agent: Optional[DataAnalystAgent] = None
news_sentiment_agent: Optional[NewsSentimentAgent] = None
report_generator_agent: Optional[ReportGeneratorAgent] = None
visualization_agent: Optional[VisualizationAgent] = None
recommendation_agent: Optional[RecommendationAgent] = None

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _ensure_directories() -> None:
    """Create required directories before the app mounts static files."""
    os.makedirs(settings.reports_output_dir, exist_ok=True)
    os.makedirs(settings.charts_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)

# Ensure dirs exist *before* mounting StaticFiles (Starlette checks at import)
_ensure_directories()

# Mount static (use check_dir=False to avoid startup crash if path changes later)
app.mount(
    "/static",
    StaticFiles(directory=settings.reports_output_dir, check_dir=False),
    name="static",
)

# ------------------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    global orchestrator_agent, data_analyst_agent, news_sentiment_agent
    global report_generator_agent, visualization_agent, recommendation_agent

    logger.info("Starting MCP-Powered Financial Analyst...")

    try:
        # DB init
        await db_manager.init_db()
        logger.info("Database initialized")

        # LLM / Model client
        model_client = create_gemini_model_client()

        # Agents
        orchestrator_agent = OrchestratorAgent(model_client)
        data_analyst_agent = DataAnalystAgent(model_client)
        news_sentiment_agent = NewsSentimentAgent(model_client)
        report_generator_agent = ReportGeneratorAgent(model_client)
        visualization_agent = VisualizationAgent(model_client)
        recommendation_agent = RecommendationAgent(model_client)

        logger.info("Agents initialized successfully")
        logger.info("Startup complete")

    except Exception as e:
        logger.exception("Failed to start application")
        # Raising ensures /health exposes 'offline' while returning 500 on requests
        raise

@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down MCP-Powered Financial Analyst...")

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP-Powered Financial Analyst</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; }}
            .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .feature-card {{ background: #ecf0f1; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db; }}
            .feature-card h3 {{ color: #2c3e50; margin-top: 0; }}
            .api-section {{ margin-top: 30px; }}
            .endpoint {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #28a745; }}
            .method {{ background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
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
                    <p>Comprehensive technical & fundamental analysis.</p>
                </div>
                <div class="feature-card">
                    <h3>📰 Sentiment Analysis</h3>
                    <p>Market/news sentiment from trusted sources.</p>
                </div>
                <div class="feature-card">
                    <h3>📈 Portfolio Optimization</h3>
                    <p>MPT-based optimization & risk controls.</p>
                </div>
                <div class="feature-card">
                    <h3>📄 Automated Reports</h3>
                    <p>Professional HTML/PDF deliverables.</p>
                </div>
                <div class="feature-card">
                    <h3>🎯 Recommendations</h3>
                    <p>Buy/Sell/Hold with rationale & confidence.</p>
                </div>
                <div class="feature-card">
                    <h3>📊 Visualizations</h3>
                    <p>Interactive charts & comparisons.</p>
                </div>
            </div>

            <div class="api-section">
                <h2>🔗 API Endpoints</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/analyze/stock</strong>
                    <p>Analyze a single stock with comprehensive metrics.</p>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/analyze/portfolio</strong>
                    <p>Analyze portfolio performance & optimization.</p>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/compare/stocks</strong>
                    <p>Compare multiple stocks side by side.</p>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/query</strong>
                    <p>Natural language financial queries.</p>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/generate/report</strong>
                    <p>Generate professional financial reports.</p>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/generate/chart</strong>
                    <p>Create stock chart visualizations.</p>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/health</strong>
                    <p>System health & agent status.</p>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/docs</strong>
                    <p>Interactive API documentation (Swagger UI).</p>
                </div>
            </div>

            <div style="margin-top: 30px; text-align: center; color: #7f8c8d;">
                <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
                <p>Built with AutoGen, FastAPI, and MCP</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "orchestrator": "online" if orchestrator_agent else "offline",
            "data_analyst": "online" if data_analyst_agent else "offline",
            "news_sentiment": "online" if news_sentiment_agent else "offline",
            "report_generator": "online" if report_generator_agent else "offline",
            "visualization": "online" if visualization_agent else "offline",
            "recommendation": "online" if recommendation_agent else "offline",
        },
        "apis": {
            "alpha_vantage": "configured" if settings.alpha_vantage_api_key else "not_configured",
            "news_api": "configured" if settings.news_api_key else "not_configured",
            "twitter_api": "configured" if settings.twitter_bearer_token else "not_configured",
        },
        "paths": {
            "reports_output_dir": os.path.abspath(settings.reports_output_dir),
            "charts_output_dir": os.path.abspath(settings.charts_output_dir),
            "log_file": os.path.abspath(settings.log_file),
        },
    }

@app.post("/api/query")
async def process_natural_language_query(request: QueryRequest) -> Dict[str, Any]:
    try:
        if not orchestrator_agent:
            raise HTTPException(status_code=500, detail="Orchestrator agent not initialized")

        logger.info("Processing query: %s", request.query)
        result = await orchestrator_agent.process_query(request.query, request.user_id)

        try:
            await db_manager.save_user_query(request.user_id, request.query, result)
        except Exception:
            logger.exception("Failed to persist user query")

        return {
            "success": True,
            "query": request.query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest) -> Dict[str, Any]:
    try:
        if not (data_analyst_agent and news_sentiment_agent and recommendation_agent):
            raise HTTPException(status_code=500, detail="Required agents not initialized")

        logger.info("Analyzing stock: %s (%s)", request.symbol, request.analysis_type)

        analysis_result = await data_analyst_agent.analyze_stock(
            request.symbol, request.analysis_type
        )
        sentiment_result = await news_sentiment_agent.analyze_stock_sentiment(request.symbol)
        recommendation_result = await recommendation_agent.analyze_stock_recommendation(analysis_result)

        return {
            "success": True,
            "symbol": request.symbol,
            "analysis": analysis_result,
            "sentiment": sentiment_result,
            "recommendation": recommendation_result,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error analyzing stock %s", request.symbol)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/portfolio")
async def analyze_portfolio(request: PortfolioAnalysisRequest) -> Dict[str, Any]:
    try:
        if not (data_analyst_agent and recommendation_agent):
            raise HTTPException(status_code=500, detail="Required agents not initialized")

        logger.info("Analyzing portfolio with %d positions", len(request.portfolio))

        portfolio_analysis = await data_analyst_agent.get_portfolio_analysis(request.portfolio)
        optimization_result = await recommendation_agent.optimize_portfolio(portfolio_analysis)
        diversification_result = await recommendation_agent.generate_diversification_recommendations(
            portfolio_analysis
        )

        return {
            "success": True,
            "portfolio_analysis": portfolio_analysis,
            "optimization": optimization_result,
            "diversification": diversification_result,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error analyzing portfolio")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare/stocks")
async def compare_stocks(request: ComparisonRequest) -> Dict[str, Any]:
    try:
        if not (data_analyst_agent and recommendation_agent and news_sentiment_agent):
            raise HTTPException(status_code=500, detail="Required agents not initialized")

        logger.info("Comparing stocks: %s", request.symbols)

        comparison_result = await data_analyst_agent.compare_stocks(request.symbols)
        investment_comparison = await recommendation_agent.compare_investment_options(comparison_result)
        sentiment_comparison = await news_sentiment_agent.compare_stock_sentiments(request.symbols)

        return {
            "success": True,
            "symbols": request.symbols,
            "comparison": comparison_result,
            "investment_analysis": investment_comparison,
            "sentiment_comparison": sentiment_comparison,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error comparing stocks")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/report")
async def generate_report(request: ReportRequest) -> Dict[str, Any]:
    try:
        if not report_generator_agent:
            raise HTTPException(status_code=500, detail="Report generator agent not initialized")

        logger.info("Generating %s report (%s)", request.analysis_type, request.format)

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
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating report")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/chart")
async def generate_chart(request: ChartRequest) -> Dict[str, Any]:
    try:
        if not (visualization_agent and data_analyst_agent):
            raise HTTPException(status_code=500, detail="Required agents not initialized")

        logger.info("Generating %s chart for %s", request.chart_type, request.symbol)

        analysis_result = await data_analyst_agent.analyze_stock(request.symbol)
        chart_result = await visualization_agent.create_stock_price_chart(
            analysis_result.get("historical_data", {}),
            request.symbol,
            request.chart_type,
        )

        return {
            "success": True,
            "chart": chart_result,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating chart for %s", request.symbol)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )

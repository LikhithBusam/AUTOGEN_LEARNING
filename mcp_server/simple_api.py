#!/usr/bin/env python3
"""
Simplified Financial Analyst API - No API Keys Required
Provides basic financial data functionality for testing
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import configuration and database
from config.settings import settings
from data.database import db_manager
from mcp.financial_data_server import mcp_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial_analyst_simple")

# FastAPI App
app = FastAPI(
    title="Financial Analyst API (Simplified)",
    description="Basic financial data API without AI agents",
    version="1.0.0-simple",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class StockRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")

class HistoricalRequest(BaseModel):
    symbol: str
    period: str = Field("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")

class ComparisonRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=2, max_items=5)

# Routes
@app.get("/")
async def root():
    return {
        "message": "Financial Analyst API (Simplified Version)",
        "status": "running",
        "features": [
            "Real-time stock prices",
            "Historical data",
            "Company information",
            "Basic stock comparison",
            "Database storage"
        ],
        "endpoints": {
            "health": "/api/health",
            "stock_price": "/api/stock/price",
            "historical": "/api/stock/historical",
            "company_info": "/api/stock/info",
            "comparison": "/api/compare"
        }
    }

@app.get("/api/health")
async def health_check():
    try:
        await db_manager.init_db()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "features": {
                "stock_prices": "available",
                "historical_data": "available", 
                "company_info": "available",
                "ai_agents": "disabled (no API keys)",
                "news_sentiment": "disabled (no API keys)"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/stock/price")
async def get_stock_price(request: StockRequest):
    """Get current stock price and basic metrics"""
    try:
        result = await mcp_client.call_method("get_stock_price", {"symbol": request.symbol})
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "symbol": request.symbol,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stock/historical")
async def get_historical_data(request: HistoricalRequest):
    """Get historical stock data"""
    try:
        result = await mcp_client.call_method("get_historical_data", {
            "symbol": request.symbol,
            "period": request.period
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stock/info")
async def get_company_info(request: StockRequest):
    """Get company information"""
    try:
        result = await mcp_client.call_method("get_company_info", {"symbol": request.symbol})
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "symbol": request.symbol,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare")
async def compare_stocks(request: ComparisonRequest):
    """Compare multiple stocks"""
    try:
        results = {}
        
        for symbol in request.symbols:
            price_data = await mcp_client.call_method("get_stock_price", {"symbol": symbol})
            company_data = await mcp_client.call_method("get_company_info", {"symbol": symbol})
            
            results[symbol] = {
                "price_data": price_data,
                "company_data": company_data
            }
        
        return {
            "success": True,
            "symbols": request.symbols,
            "comparison": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Financial Analyst API (Simplified)...")
    try:
        await db_manager.init_db()
        logger.info("Database initialized")
        logger.info("API ready - No API keys required for basic functionality")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Starting Financial Analyst API (Simplified)")
    print("📊 This version provides basic financial data without requiring API keys")
    print("🌐 Open http://localhost:8000 in your browser")
    print("📖 API documentation: http://localhost:8000/docs")
    print("💚 Health check: http://localhost:8000/api/health")
    print()
    print("Example usage:")
    print("  POST /api/stock/price with {\"symbol\": \"AAPL\"}")
    print("  POST /api/stock/historical with {\"symbol\": \"AAPL\", \"period\": \"1mo\"}")
    print("  POST /api/stock/info with {\"symbol\": \"AAPL\"}")
    print("  POST /api/compare with {\"symbols\": [\"AAPL\", \"MSFT\", \"GOOGL\"]}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

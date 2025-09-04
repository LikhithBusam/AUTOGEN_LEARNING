#!/usr/bin/env python3
"""
Simple API server test without agents
"""
import sys
import os
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Create a simple test API
app = FastAPI(title="Financial Analyst Test API")

@app.get("/")
async def root():
    return {"message": "Financial Analyst API is running", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    try:
        from config.settings import settings
        from data.database import db_manager
        
        # Test database connection
        await db_manager.init_db()
        
        return {
            "status": "healthy",
            "database": "connected",
            "config": "loaded",
            "reports_dir": settings.reports_output_dir
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/api/test/stock/{symbol}")
async def test_stock_data(symbol: str):
    try:
        from mcp.financial_data_server import mcp_client
        
        result = await mcp_client.call_method("get_stock_price", {"symbol": symbol})
        return {"symbol": symbol, "data": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    print("🚀 Starting Financial Analyst Test API Server...")
    print("🌐 Open http://localhost:8001 in your browser")
    print("📊 Test endpoint: http://localhost:8001/api/test/stock/AAPL")
    print("💚 Health check: http://localhost:8001/api/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

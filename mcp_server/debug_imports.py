"""
Debug imports for MCP financial data server
"""
import sys
import traceback

print("🔍 Testing imports step by step...")

try:
    print("1. Testing basic imports...")
    import asyncio
    import json
    from typing import Any, Dict, List, Optional
    from datetime import datetime, timedelta
    import yfinance as yf
    import httpx
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    from functools import partial
    print("   ✅ Basic imports successful")
except Exception as e:
    print(f"   ❌ Basic imports failed: {e}")
    traceback.print_exc()

try:
    print("2. Testing config import...")
    from config.settings import settings
    print("   ✅ Settings import successful")
except Exception as e:
    print(f"   ❌ Settings import failed: {e}")
    traceback.print_exc()

try:
    print("3. Testing database import...")
    from data.database import db_manager
    print("   ✅ Database import successful")
except Exception as e:
    print(f"   ❌ Database import failed: {e}")
    traceback.print_exc()

try:
    print("4. Testing MCP server class import...")
    from mcp.financial_data_server import FinancialDataMCPServer
    print("   ✅ MCP server class import successful")
except Exception as e:
    print(f"   ❌ MCP server class import failed: {e}")
    traceback.print_exc()

try:
    print("5. Testing MCP client import...")
    from mcp.financial_data_server import mcp_client
    print("   ✅ MCP client import successful")
except Exception as e:
    print(f"   ❌ MCP client import failed: {e}")
    traceback.print_exc()

print("\n🎯 Import debug complete!")

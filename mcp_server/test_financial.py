#!/usr/bin/env python3
"""
Test financial data functionality without API keys
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_financial_data():
    print("Testing financial data functionality...")
    
    try:
        from mcp.financial_data_server import mcp_client
        print("✅ MCP client imported")
        
        # Test basic stock price retrieval (uses yfinance, no API key needed)
        result = await mcp_client.call_method("get_stock_price", {"symbol": "AAPL"})
        
        if "error" in result:
            print(f"⚠️ Stock price retrieval returned error: {result['error']}")
        else:
            print("✅ Stock price retrieval working")
            print(f"   AAPL current price: ${result.get('current_price', 'N/A')}")
        
        # Test company info
        company_result = await mcp_client.call_method("get_company_info", {"symbol": "AAPL"})
        
        if "error" in company_result:
            print(f"⚠️ Company info retrieval returned error: {company_result['error']}")
        else:
            print("✅ Company info retrieval working")
            print(f"   Company: {company_result.get('company_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Financial data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_financial_data())
    if result:
        print("🎉 Financial data test completed!")
    else:
        print("⚠️ Financial data test failed!")

#!/usr/bin/env python3
"""
Test client for the Financial Analyst API
"""
import asyncio
import httpx
import json

API_BASE = "http://localhost:8000"

async def test_api():
    """Test the Financial Analyst API endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Financial Analyst API")
        print("=" * 50)
        
        # Test 1: Health Check
        print("\n1️⃣  Testing Health Check...")
        try:
            response = await client.get(f"{API_BASE}/api/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Database: {data.get('database')}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test 2: Stock Price
        print("\n2️⃣  Testing Stock Price (AAPL)...")
        try:
            response = await client.post(
                f"{API_BASE}/api/stock/price",
                json={"symbol": "AAPL"}
            )
            if response.status_code == 200:
                print("✅ Stock price retrieval successful")
                data = response.json()
                price_data = data.get("data", {})
                print(f"   AAPL Price: ${price_data.get('current_price', 'N/A')}")
                print(f"   Change: {price_data.get('change_percent', 'N/A'):.2f}%" if price_data.get('change_percent') else "   Change: N/A")
            else:
                print(f"❌ Stock price failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Stock price error: {e}")
        
        # Test 3: Company Info
        print("\n3️⃣  Testing Company Info (MSFT)...")
        try:
            response = await client.post(
                f"{API_BASE}/api/stock/info",
                json={"symbol": "MSFT"}
            )
            if response.status_code == 200:
                print("✅ Company info retrieval successful")
                data = response.json()
                company_data = data.get("data", {})
                print(f"   Company: {company_data.get('company_name', 'N/A')}")
                print(f"   Sector: {company_data.get('sector', 'N/A')}")
                print(f"   Market Cap: {company_data.get('market_cap', 'N/A')}")
            else:
                print(f"❌ Company info failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Company info error: {e}")
        
        # Test 4: Historical Data
        print("\n4️⃣  Testing Historical Data (GOOGL - 1 month)...")
        try:
            response = await client.post(
                f"{API_BASE}/api/stock/historical",
                json={"symbol": "GOOGL", "period": "1mo"}
            )
            if response.status_code == 200:
                print("✅ Historical data retrieval successful")
                data = response.json()
                hist_data = data.get("data", {})
                data_points = hist_data.get("data", [])
                print(f"   Data points: {len(data_points)}")
                if data_points:
                    latest = data_points[-1]
                    print(f"   Latest close: ${latest.get('close', 'N/A')}")
            else:
                print(f"❌ Historical data failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Historical data error: {e}")
        
        # Test 5: Stock Comparison
        print("\n5️⃣  Testing Stock Comparison (AAPL vs MSFT vs GOOGL)...")
        try:
            response = await client.post(
                f"{API_BASE}/api/compare",
                json={"symbols": ["AAPL", "MSFT", "GOOGL"]}
            )
            if response.status_code == 200:
                print("✅ Stock comparison successful")
                data = response.json()
                comparison = data.get("comparison", {})
                print(f"   Compared {len(comparison)} stocks:")
                for symbol, stock_data in comparison.items():
                    price = stock_data.get("price_data", {}).get("current_price")
                    company = stock_data.get("company_data", {}).get("company_name")
                    print(f"     {symbol}: ${price:.2f} - {company}" if price and company else f"     {symbol}: Data unavailable")
            else:
                print(f"❌ Stock comparison failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Stock comparison error: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 API testing completed!")
        print("\n📖 For full API documentation, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running on localhost:8000")
    print()
    
    try:
        asyncio.run(test_api())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")

#!/usr/bin/env python3
"""
Test client for the Intelligent Financial Query System
"""
import asyncio
import httpx
import json

API_BASE = "http://localhost:5000"

async def test_intelligent_queries():
    """Test the intelligent financial query system"""
    
    # Test queries to demonstrate AI capabilities
    test_queries = [
        "How is Apple stock performing today?",
        "Compare Tesla vs Ford stocks for investment",
        "AAPL vs MSFT analysis",
        "Should I buy Amazon stock?",
        "What is P/E ratio and why is it important?",
        "How to start investing as a beginner?",
        "Market overview today",
        "Portfolio diversification strategies",
        "Risk assessment for NVIDIA stock",
        "Best tech stocks to invest in"
    ]
    
    async with httpx.AsyncClient() as client:
        print("🤖 Testing Intelligent Financial Query System")
        print("=" * 60)
        
        # Test health endpoint first
        print("\n🏥 Health Check...")
        try:
            response = await client.get(f"{API_BASE}/api/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ System Status: {health.get('status')}")
                print(f"   Components: {health.get('components', {})}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("   Make sure the Flask app is running on localhost:5000")
            return
        
        # Test intelligent queries
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}️⃣  Query: '{query}'")
            print("-" * 50)
            
            try:
                response = await client.post(
                    f"{API_BASE}/api/intelligent-query",
                    json={"query": query, "user_id": "test_user"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✅ Success: {result.get('success')}")
                    print(f"📊 Query Type: {result.get('query_type')}")
                    
                    if result.get('detected_symbols'):
                        print(f"🔍 Detected Symbols: {result.get('detected_symbols')}")
                    
                    # Show human response
                    human_response = result.get('response', '')
                    if human_response:
                        print(f"\n🤖 AI Response:")
                        print(human_response)
                    
                    # Show any errors
                    if result.get('error'):
                        print(f"⚠️  Error: {result.get('error')}")
                
                else:
                    print(f"❌ Request failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
            
            # Small delay between queries
            await asyncio.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("🎉 Intelligent Query Testing Completed!")
        print("\n📖 API Documentation:")
        print(f"   Health: {API_BASE}/api/health")
        print(f"   Intelligent Query: POST {API_BASE}/api/intelligent-query")
        print(f"   Regular Query: POST {API_BASE}/api/query")
        print(f"   Test Queries: {API_BASE}/api/test-queries")
        print(f"   Stock Analysis: {API_BASE}/api/analyze/stock/<symbol>")
        print(f"   Stock Comparison: POST {API_BASE}/api/compare/stocks")

if __name__ == "__main__":
    print("🚀 Starting Intelligent Query System Tests...")
    print("   Make sure your Flask app is running: python app.py")
    print()
    
    try:
        asyncio.run(test_intelligent_queries())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")

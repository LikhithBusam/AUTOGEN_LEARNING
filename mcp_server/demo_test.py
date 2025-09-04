"""
Demo script to test the MCP-Powered Financial Analyst APIs
"""
import requests
import json
import time

def test_api_endpoints():
    """Test the main API endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🚀 Testing MCP-Powered Financial Analyst APIs")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check Successful")
            print(f"   Status: {data['status']}")
            print(f"   Agents: {list(data['agents'].keys())}")
            print(f"   APIs: {data['apis']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
    
    # Test 2: Stock Analysis
    print("\n2. Testing Stock Analysis...")
    try:
        payload = {
            "symbol": "AAPL",
            "analysis_type": "comprehensive"
        }
        response = requests.post(f"{base_url}/api/analyze/stock", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Stock Analysis Successful")
            print(f"   Symbol: {data['symbol']}")
            print(f"   Analysis Keys: {list(data['analysis'].keys()) if 'analysis' in data else 'None'}")
        else:
            print(f"❌ Stock Analysis Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Stock Analysis Error: {e}")
    
    # Test 3: Natural Language Query
    print("\n3. Testing Natural Language Query...")
    try:
        payload = {
            "query": "What is the current price of Apple stock?",
            "user_id": "test_user"
        }
        response = requests.post(f"{base_url}/api/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Query Processing Successful")
            print(f"   Query: {data['query']}")
            print(f"   Result Keys: {list(data['result'].keys()) if 'result' in data else 'None'}")
        else:
            print(f"❌ Query Processing Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Query Processing Error: {e}")
    
    # Test 4: Market Sentiment
    print("\n4. Testing Market Sentiment...")
    try:
        response = requests.get(f"{base_url}/api/sentiment/market")
        if response.status_code == 200:
            data = response.json()
            print("✅ Market Sentiment Successful")
            print(f"   Sentiment: {data.get('market_sentiment', {})}")
        else:
            print(f"❌ Market Sentiment Failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Market Sentiment Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Testing Complete!")
    print("Visit http://localhost:8000 for the web interface")
    print("Visit http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    test_api_endpoints()

#!/usr/bin/env python3
"""
Test client for the Intelligent Financial Assistant
Demonstrates the AI-powered query processing capabilities
"""
import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_intelligent_queries():
    """Test various types of financial queries"""
    
    test_queries = [
        # Single stock analysis
        {
            "query": "How is Apple stock performing today?",
            "expected_type": "single_stock_analysis"
        },
        {
            "query": "Should I invest in Tesla?",
            "expected_type": "single_stock_analysis"
        },
        {
            "query": "What's the current price of Microsoft?",
            "expected_type": "single_stock_analysis"
        },
        
        # Stock comparison
        {
            "query": "Compare Apple vs Microsoft stocks for long-term investment",
            "expected_type": "multi_stock_comparison"
        },
        {
            "query": "AAPL vs GOOGL vs AMZN performance analysis",
            "expected_type": "multi_stock_comparison"
        },
        {
            "query": "Which is better: Tesla or Ford?",
            "expected_type": "multi_stock_comparison"
        },
        
        # General market queries
        {
            "query": "What are the best tech stocks to buy?",
            "expected_type": "general_market_with_suggestions"
        },
        {
            "query": "Tell me about the current market trends",
            "expected_type": "general_market_with_suggestions"
        },
        
        # Intent-specific queries
        {
            "query": "Give me investment recommendations for retirement",
            "expected_type": "general_market_with_suggestions"
        },
        {
            "query": "What's the news sentiment around NVDA?",
            "expected_type": "single_stock_analysis"
        }
    ]
    
    print("🤖 Testing Intelligent Financial Assistant")
    print("=" * 60)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_type = test_case["expected_type"]
        
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 50)
        
        try:
            response = requests.post(
                f"{API_BASE}/api/query",
                json={"query": query, "user_id": f"test_user_{i}"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key information
                query_type = result.get("query_type", "unknown")
                success = result.get("success", False)
                summary = result.get("summary", "No summary available")
                
                print(f"✅ Status: {'Success' if success else 'Failed'}")
                print(f"📊 Query Type: {query_type}")
                print(f"🎯 Expected: {expected_type}")
                print(f"✨ Match: {'Yes' if query_type == expected_type else 'No'}")
                
                # Show detected symbols if available
                if "symbols" in result:
                    print(f"🏷️  Symbols: {result['symbols']}")
                elif "detected_symbols" in result:
                    print(f"🏷️  Detected Symbols: {result['detected_symbols']}")
                
                # Show summary (truncated)
                if summary and len(summary) > 200:
                    summary = summary[:200] + "..."
                print(f"💬 Summary: {summary}")
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
        
        time.sleep(1)  # Brief pause between requests
    
    print("\n" + "=" * 60)
    print("🎉 Intelligent Assistant Testing Complete!")

def test_specific_features():
    """Test specific features of the intelligent assistant"""
    
    print("\n🔧 Testing Specific Features")
    print("=" * 40)
    
    # Test 1: Symbol extraction
    print("\n1️⃣  Testing Symbol Extraction:")
    symbol_queries = [
        "Tell me about Apple and Microsoft",
        "Compare AAPL with GOOGL",
        "How are Tesla, Ford, and GM doing?",
        "I want to know about Nvidia stock"
    ]
    
    for query in symbol_queries:
        try:
            response = requests.post(
                f"{API_BASE}/api/query",
                json={"query": query},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                symbols = result.get("symbols", result.get("detected_symbols", []))
                print(f"   '{query}' → {symbols}")
            
        except Exception as e:
            print(f"   Error testing: {query}")
    
    # Test 2: Intent detection
    print("\n2️⃣  Testing Intent Detection:")
    intent_queries = [
        ("Compare Apple vs Google", "comparison"),
        ("Should I buy Tesla?", "recommendation"),
        ("What's the price of Amazon?", "price"),
        ("How is Netflix performing?", "performance")
    ]
    
    for query, expected_intent in intent_queries:
        try:
            response = requests.post(
                f"{API_BASE}/api/query",
                json={"query": query},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                detected_intent = result.get("detected_intent", "unknown")
                print(f"   '{query}' → {detected_intent} (expected: {expected_intent})")
            
        except Exception as e:
            print(f"   Error testing intent: {query}")

def test_api_health():
    """Test if the API is running and healthy"""
    
    print("🏥 Testing API Health")
    print("-" * 30)
    
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy")
            print(f"   Status: {health_data.get('status')}")
            
            components = health_data.get('components', {})
            for component, status in components.items():
                print(f"   {component}: {status}")
            
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        print("   Make sure the Flask app is running on localhost:5000")
        return False

if __name__ == "__main__":
    print("🚀 Starting Intelligent Financial Assistant Tests")
    print("📡 Connecting to API at http://localhost:5000")
    print()
    
    # Check API health first
    if test_api_health():
        print()
        test_intelligent_queries()
        test_specific_features()
    else:
        print("\n⚠️  Please start the Flask application first:")
        print("   python app.py")

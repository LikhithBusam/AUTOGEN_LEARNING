#!/usr/bin/env python3
"""
Simple test to verify portfolio detection
"""

def test_simple():
    print("🔍 TESTING PORTFOLIO DETECTION")
    
    query = "Create a diversified portfolio recommendation for $10,000"
    query_lower = query.lower()
    
    # Test our detection logic
    portfolio_keywords = ['portfolio', 'holdings', 'allocation', 'diversification', 'diversified', 'invest', 'allocate', 'build a portfolio', 'recommendation for']
    
    detected = any(word in query_lower for word in portfolio_keywords)
    
    print(f"Query: {query}")
    print(f"Query Lower: {query_lower}")
    print(f"Keywords: {portfolio_keywords}")
    print(f"Detection Result: {detected}")
    
    for keyword in portfolio_keywords:
        if keyword in query_lower:
            print(f"✅ Found keyword: '{keyword}'")
    
    if detected:
        print("🎉 PORTFOLIO DETECTION SHOULD WORK!")
    else:
        print("❌ Detection failed")

if __name__ == "__main__":
    test_simple()

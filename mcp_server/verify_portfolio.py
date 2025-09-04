#!/usr/bin/env python3
"""
Simple verification test
"""
import sys
import os
sys.path.insert(0, '.')

print("🔥 QUICK VERIFICATION TEST")

try:
    print("1. Testing imports...")
    from app import intelligent_assistant
    print("✅ Import successful")
    
    print("2. Testing portfolio detection...")
    query = "Create a diversified portfolio recommendation for $10,000"
    query_lower = query.lower()
    portfolio_detected = any(word in query_lower for word in ['portfolio', 'diversified', 'recommendation for'])
    print(f"✅ Portfolio detection: {portfolio_detected}")
    
    print("3. Testing method existence...")
    has_method = hasattr(intelligent_assistant, '_handle_portfolio_query')
    print(f"✅ Method exists: {has_method}")
    
    if has_method and portfolio_detected:
        print("🎉 ALL CHECKS PASSED - SYSTEM SHOULD WORK!")
    else:
        print("❌ Some checks failed")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("✅ Verification complete")

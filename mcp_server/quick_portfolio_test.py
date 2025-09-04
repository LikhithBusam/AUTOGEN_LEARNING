#!/usr/bin/env python3
"""
Quick test for portfolio functionality
"""
import sys
import os
sys.path.insert(0, '.')

def test_portfolio_quick():
    print("💼 QUICK PORTFOLIO TEST")
    print("=" * 40)
    
    try:
        from app import intelligent_assistant
        
        # Test a simple portfolio query
        result = intelligent_assistant.process_query("Create a diversified portfolio recommendation for $10,000")
        
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📈 Query Type: {result.get('query_type', 'Unknown')}")
        
        if result.get('query_type') == 'portfolio_recommendation':
            print("🎯 PORTFOLIO DETECTED!")
            
            # Show sentence format
            sentence = result.get('sentence_format', '')
            print(f"💬 Response: {sentence[:150]}...")
            
            # Show investment amount
            amount = result.get('investment_amount', 0)
            if amount:
                print(f"💵 Investment Amount: ${amount:,.0f}")
            
            print("🎉 PORTFOLIO SYSTEM WORKING!")
        else:
            print(f"❌ Wrong detection: {result.get('query_type')}")
            sentence = result.get('sentence_format', '')
            if sentence:
                print(f"💬 Got: {sentence[:100]}...")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_portfolio_quick()

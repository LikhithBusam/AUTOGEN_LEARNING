#!/usr/bin/env python3
"""
Final test to verify the complete portfolio system
"""

def main():
    print("🚀 FINAL PORTFOLIO SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Import and test
        print("📦 Importing system...")
        from app import intelligent_assistant
        
        if not intelligent_assistant:
            print("❌ System not available")
            return
        
        print("✅ System imported successfully")
        
        # Test the exact query
        query = "Create a diversified portfolio recommendation for $10,000"
        print(f"\n🎯 Testing query: {query}")
        
        result = intelligent_assistant.process_query(query)
        
        print(f"\n📊 RESULTS:")
        print(f"Success: {result.get('success', False)}")
        print(f"Query Type: {result.get('query_type', 'Unknown')}")
        
        # Check if it's correctly detected as portfolio
        if result.get('query_type') == 'portfolio_recommendation':
            print("🎉 SUCCESS! Portfolio detected correctly!")
            
            # Show investment amount
            amount = result.get('investment_amount', 0)
            if amount:
                print(f"💰 Investment Amount: ${amount:,.0f}")
            
            # Show sentence format
            sentence = result.get('sentence_format', '')
            if sentence:
                print(f"📝 Response: {sentence[:100]}...")
            
            # Show if we have analysis
            analysis = result.get('analysis', {})
            if 'portfolio_summary' in analysis:
                summary = analysis['portfolio_summary']
                print(f"📈 Positions: {summary.get('total_positions', 0)}")
                print(f"🏢 Sectors: {summary.get('sectors_covered', 0)}")
                print(f"💵 Invested: ${summary.get('total_invested', 0):,.0f}")
                print("✅ COMPLETE PORTFOLIO SYSTEM WORKING!")
            else:
                print("⚠️  Basic portfolio guidance provided")
        
        elif result.get('query_type') == 'portfolio':
            print("✅ Portfolio detected (basic guidance)")
            
        else:
            print(f"❌ WRONG! Detected as: {result.get('query_type')}")
            print("Portfolio detection still not working correctly")
            
            # Show what we got instead
            sentence = result.get('sentence_format', '')
            if sentence:
                print(f"Got: {sentence[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

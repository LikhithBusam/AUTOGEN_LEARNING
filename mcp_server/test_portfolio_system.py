#!/usr/bin/env python3
"""
Test the enhanced portfolio recommendation system
"""
import sys
import os
sys.path.insert(0, '.')

def test_portfolio_recommendation():
    print("💼 TESTING PORTFOLIO RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    try:
        from app import intelligent_assistant
        
        if not intelligent_assistant:
            print("❌ System not available")
            return
            
        # Test portfolio queries
        portfolio_queries = [
            "Create a diversified portfolio recommendation for $10,000",
            "Build a portfolio for $25,000",
            "How should I allocate $50,000 in stocks?",
            "Portfolio recommendation for $5,000"
        ]
        
        for i, query in enumerate(portfolio_queries, 1):
            print(f"\n💰 Test {i}: {query}")
            print("-" * 40)
            
            result = intelligent_assistant.process_query(query)
            
            print(f"✅ Success: {result.get('success', False)}")
            print(f"📊 Query Type: {result.get('query_type', 'Unknown')}")
            
            # Check for proper portfolio response
            if result.get('query_type') == 'portfolio_recommendation':
                print("🎯 PORTFOLIO RECOMMENDATION DETECTED!")
                
                # Show sentence format
                sentence = result.get('sentence_format', '')
                if sentence:
                    print(f"💬 Summary: {sentence}")
                
                # Show investment amount
                amount = result.get('investment_amount', 0)
                if amount:
                    print(f"💵 Investment Amount: ${amount:,.0f}")
                
                # Show table format (truncated)
                table = result.get('table_format', '')
                if table:
                    lines = table.split('\n')
                    print("📋 Portfolio Allocation (First 5 rows):")
                    for line in lines[:7]:  # Header + first few rows
                        print(f"   {line}")
                    if len(lines) > 7:
                        print("   ... (more positions)")
                
                # Show analysis
                analysis = result.get('analysis', {})
                if 'portfolio_summary' in analysis:
                    summary = analysis['portfolio_summary']
                    print(f"📈 Total Positions: {summary.get('total_positions', 0)}")
                    print(f"🏢 Sectors Covered: {summary.get('sectors_covered', 0)}")
                    print(f"💰 Total Invested: ${summary.get('total_invested', 0):,.0f}")
                    print(f"💸 Cash Remaining: ${summary.get('cash_remaining', 0):,.0f}")
                    
                    ai_analysis = summary.get('ai_analysis', '')
                    if ai_analysis:
                        print(f"🤖 AI Analysis: {ai_analysis[:100]}...")
            
            elif result.get('query_type') == 'portfolio':
                print("📚 Basic portfolio guidance provided")
                
            else:
                print(f"⚠️  Wrong query type detected: {result.get('query_type')}")
                print("❌ Portfolio detection may need improvement")
            
            print("=" * 60)
        
        print("\n🎉 PORTFOLIO RECOMMENDATION TEST COMPLETE!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_portfolio_recommendation()

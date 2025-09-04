#!/usr/bin/env python3
"""
Test the AI-powered intelligent responses vs rule-based responses
"""

import sys
import os
sys.path.insert(0, '.')

def test_ai_vs_static():
    """Compare AI responses with different stocks to show dynamic analysis"""
    
    print("🤖 TESTING AI-POWERED vs STATIC RESPONSES")
    print("=" * 60)
    
    test_stocks = ["AAPL", "TSLA", "NVDA", "MSFT"]
    
    try:
        from app import intelligent_assistant
        
        if not intelligent_assistant:
            print("❌ Intelligent assistant not available")
            return
        
        for i, stock in enumerate(test_stocks, 1):
            print(f"\n📊 Test {i}: Analyzing {stock}")
            print("-" * 40)
            
            query = f"Should I invest in {stock}?"
            result = intelligent_assistant.process_query(query)
            
            if result.get('success'):
                analysis = result.get('analysis', {})
                recommendation = analysis.get('recommendation', {})
                
                # Show key AI-generated insights
                print(f"🎯 AI Recommendation: {recommendation.get('action', 'N/A')}")
                print(f"🎓 Confidence: {recommendation.get('confidence', 0)*100:.0f}%")
                print(f"📈 Analysis Method: {analysis.get('analysis_method', 'Unknown')}")
                
                # Show AI reasoning if available
                reasoning = recommendation.get('reasoning', [])
                if reasoning:
                    print(f"🧠 AI Reasoning:")
                    for reason in reasoning[:3]:  # Show first 3 reasons
                        print(f"   • {reason}")
                
                # Show if it's using real AI vs static rules
                if 'market_outlook' in recommendation:
                    print(f"🌍 Market Outlook: {recommendation.get('market_outlook', 'N/A')}")
                
                if 'key_strengths' in recommendation:
                    strengths = recommendation.get('key_strengths', [])
                    if strengths:
                        print(f"💪 Key Strengths: {', '.join(strengths[:2])}")
                
                # Show sentence format (human-readable)
                sentence = result.get('sentence_format', '')
                if sentence:
                    print(f"💬 AI Summary: {sentence[:150]}...")
                    
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            print("=" * 60)
        
        print("\n🎯 ANALYSIS SUMMARY:")
        print("If you see different recommendations, reasoning, and market outlooks")
        print("for different stocks, then the AI is working dynamically! 🎉")
        print("\nIf all recommendations are similar (like 60% confidence, similar")
        print("reasoning), then it's using static rule-based fallback. 📊")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_ai_vs_static()

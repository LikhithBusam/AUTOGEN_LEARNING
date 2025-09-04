#!/usr/bin/env python3
"""
Test the news sentiment analysis functionality
"""

import sys
import os
sys.path.insert(0, '.')

def test_news_functionality():
    """Test the news and sentiment analysis features"""
    
    print("📰 TESTING NEWS SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        from app import intelligent_assistant, news_analyzer
        
        if not intelligent_assistant:
            print("❌ Intelligent assistant not available")
            return
        
        # Test different news queries
        test_queries = [
            "Get latest news for AAPL",
            "What's the news sentiment for Tesla?", 
            "Fetch latest news for any stock",
            "News analysis for Microsoft",
            "Market news today"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📊 Test {i}: {query}")
            print("-" * 40)
            
            # Test through intelligent assistant
            result = intelligent_assistant.process_query(query)
            
            print(f"✅ Success: {result.get('success', False)}")
            print(f"📈 Query Type: {result.get('query_type', 'Unknown')}")
            print(f"🎯 Intent: {result.get('detected_intent', 'Unknown')}")
            
            # Check if news data is present
            if 'news_data' in result.get('analysis', {}):
                news_data = result['analysis']['news_data']
                articles = news_data.get('articles', [])
                sentiment = news_data.get('overall_sentiment', {})
                
                print(f"📰 Articles Found: {len(articles)}")
                print(f"😊 Overall Sentiment: {sentiment.get('label', 'unknown')} ({sentiment.get('score', 0):.2f})")
                
                if articles:
                    print(f"📝 Latest Headline: {articles[0].get('title', 'N/A')[:60]}...")
            
            # Check sentence format
            sentence = result.get('sentence_format', '')
            if sentence:
                print(f"💬 Sentence: {sentence[:80]}...")
            
            # Check for errors
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            
            print("=" * 60)
        
        # Test direct news analyzer
        print(f"\n🔬 TESTING DIRECT NEWS ANALYZER")
        print("-" * 40)
        
        try:
            direct_news = news_analyzer.get_stock_news("AAPL", limit=3)
            print(f"✅ Direct news fetch success: {direct_news.get('success', False)}")
            
            if direct_news.get('success'):
                articles = direct_news.get('articles', [])
                print(f"📰 Articles fetched: {len(articles)}")
                
                if articles:
                    for i, article in enumerate(articles, 1):
                        sentiment = article.get('sentiment', {})
                        print(f"{i}. {sentiment.get('label', 'neutral')} - {article.get('title', '')[:50]}...")
            
        except Exception as e:
            print(f"❌ Direct news test failed: {e}")
        
        print("\n🎉 News Analysis Test Complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_news_functionality()

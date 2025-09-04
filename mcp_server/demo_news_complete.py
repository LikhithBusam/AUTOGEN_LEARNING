#!/usr/bin/env python3
"""
Comprehensive news sentiment analysis demo
"""
import sys
import os
sys.path.insert(0, '.')

def demo_news_capabilities():
    print("🌟 INTELLIGENT FINANCIAL ASSISTANT - NEWS DEMO")
    print("=" * 60)
    
    try:
        from app import intelligent_assistant, news_analyzer
        
        if not intelligent_assistant:
            print("❌ System not available")
            return
            
        print("🤖 AI-Powered Financial Analysis System Ready!")
        print("📰 News Sentiment Analysis Capabilities")
        print("-" * 60)
        
        # Demo queries
        demo_queries = [
            ("Apple News", "Get latest news for Apple stock"),
            ("Tesla Sentiment", "What's the sentiment for Tesla?"),
            ("Market News", "Show me today's market news"),
            ("Microsoft Analysis", "Analyze Microsoft news sentiment"),
            ("General News", "Fetch financial news")
        ]
        
        for title, query in demo_queries:
            print(f"\n📊 {title}")
            print("-" * 30)
            print(f"Query: {query}")
            
            # Process the query
            result = intelligent_assistant.process_query(query)
            
            if result.get('success'):
                # Show the AI-generated response
                sentence = result.get('sentence_format', '')
                if sentence:
                    print(f"🤖 AI Response: {sentence[:150]}...")
                
                # Show detected query type
                query_type = result.get('query_type', 'unknown')
                print(f"🎯 Detected as: {query_type}")
                
                # Show if news data was found
                analysis = result.get('analysis', {})
                if 'news_data' in analysis:
                    news_data = analysis['news_data']
                    articles = news_data.get('articles', [])
                    sentiment = news_data.get('overall_sentiment', {})
                    
                    print(f"📰 News Articles: {len(articles)} found")
                    
                    if sentiment:
                        sentiment_label = sentiment.get('label', 'neutral')
                        sentiment_score = sentiment.get('score', 0.0)
                        print(f"😊 Overall Sentiment: {sentiment_label} ({sentiment_score:.2f})")
                    
                    # Show sample headlines
                    if articles:
                        print("📝 Sample Headlines:")
                        for i, article in enumerate(articles[:2], 1):
                            title = article.get('title', 'No title')
                            article_sentiment = article.get('sentiment', {})
                            sentiment_emoji = {
                                'positive': '📈',
                                'negative': '📉', 
                                'neutral': '📊'
                            }.get(article_sentiment.get('label', 'neutral'), '📊')
                            
                            print(f"   {sentiment_emoji} {title[:60]}...")
                
                print("✅ Success")
            else:
                print("❌ Query failed")
                if 'error' in result:
                    print(f"Error: {result['error']}")
            
            print("=" * 60)
        
        # Show direct news analyzer capabilities
        print(f"\n🔬 DIRECT NEWS ANALYZER DEMO")
        print("-" * 40)
        
        try:
            # Test direct news fetching
            news_result = news_analyzer.get_stock_news("AAPL", limit=2)
            
            if news_result.get('success'):
                articles = news_result.get('articles', [])
                print(f"✅ Fetched {len(articles)} articles directly")
                
                for i, article in enumerate(articles, 1):
                    title = article.get('title', 'No title')
                    sentiment = article.get('sentiment', {})
                    source = article.get('source', 'Unknown')
                    
                    print(f"{i}. [{source}] {title[:50]}...")
                    print(f"   Sentiment: {sentiment.get('label', 'neutral')} ({sentiment.get('score', 0):.2f})")
            else:
                print("❌ Direct news fetch failed")
                
        except Exception as e:
            print(f"❌ Direct test error: {e}")
        
        print("\n🎉 NEWS SENTIMENT ANALYSIS DEMO COMPLETE!")
        print("🌟 Your AI-powered financial assistant is ready!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_news_capabilities()

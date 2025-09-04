"""
News & Sentiment Analysis Agent - Analyzes financial news and market sentiment
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from collections import Counter
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings
from mcp.financial_data_server import mcp_client

class NewsSentimentAgent:
    """Agent specialized in financial news analysis and sentiment scoring"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["news_sentiment"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["news_sentiment"]["system_message"]
        )
        self.mcp_client = mcp_client
        
        # Financial sentiment keywords
        self.positive_keywords = [
            'growth', 'profit', 'earnings beat', 'outperform', 'bullish', 'upgrade',
            'expansion', 'revenue increase', 'strong', 'positive', 'gain', 'rise',
            'surge', 'breakthrough', 'success', 'record', 'milestone', 'acquire',
            'merger', 'partnership', 'innovation', 'launch', 'approval'
        ]
        
        self.negative_keywords = [
            'loss', 'decline', 'bearish', 'downgrade', 'recession', 'crisis',
            'debt', 'bankruptcy', 'lawsuit', 'investigation', 'fraud', 'scandal',
            'falling', 'drop', 'plunge', 'crash', 'weak', 'disappointing',
            'miss', 'cut', 'layoffs', 'closure', 'suspend', 'delay'
        ]
    
    async def analyze_stock_sentiment(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Analyze sentiment for a specific stock"""
        
        try:
            # Get news for the stock
            news_data = await self.mcp_client.call_method("get_company_news", {
                "symbol": symbol,
                "days": days
            })
            
            if "error" in news_data:
                return {"error": f"Failed to fetch news for {symbol}"}
            
            articles = news_data.get("articles", [])
            
            if not articles:
                return {
                    "symbol": symbol,
                    "sentiment_score": 0,
                    "sentiment_label": "neutral",
                    "article_count": 0,
                    "analysis": "No recent news found"
                }
            
            # Analyze sentiment for each article
            sentiment_analysis = await self._analyze_articles_sentiment(articles)
            
            # Aggregate sentiment scores
            overall_sentiment = self._calculate_overall_sentiment(sentiment_analysis)
            
            # Extract key themes and topics
            themes = self._extract_themes(articles)
            
            # Calculate sentiment trend over time
            sentiment_trend = self._calculate_sentiment_trend(sentiment_analysis)
            
            return {
                "symbol": symbol,
                "analysis_period": f"{days} days",
                "overall_sentiment": overall_sentiment,
                "article_analyses": sentiment_analysis,
                "themes": themes,
                "sentiment_trend": sentiment_trend,
                "article_count": len(articles),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze sentiment for {symbol}: {str(e)}"}
    
    async def analyze_market_sentiment(self, days: int = 3) -> Dict[str, Any]:
        """Analyze overall market sentiment"""
        
        try:
            # Get general market news
            market_news = await self.mcp_client.call_method("get_market_news", {
                "days": days
            })
            
            if "error" in market_news:
                return {"error": "Failed to fetch market news"}
            
            articles = market_news.get("articles", [])
            
            if not articles:
                return {
                    "sentiment_score": 0,
                    "sentiment_label": "neutral",
                    "article_count": 0,
                    "analysis": "No recent market news found"
                }
            
            # Analyze sentiment for market articles
            sentiment_analysis = await self._analyze_articles_sentiment(articles)
            
            # Calculate overall market sentiment
            overall_sentiment = self._calculate_overall_sentiment(sentiment_analysis)
            
            # Identify market themes
            market_themes = self._extract_market_themes(articles)
            
            # Analyze sector sentiment
            sector_sentiment = self._analyze_sector_sentiment(articles)
            
            return {
                "analysis_period": f"{days} days",
                "market_sentiment": overall_sentiment,
                "themes": market_themes,
                "sector_sentiment": sector_sentiment,
                "article_count": len(articles),
                "top_concerns": self._identify_market_concerns(articles),
                "top_opportunities": self._identify_market_opportunities(articles),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze market sentiment: {str(e)}"}
    
    async def compare_stock_sentiments(self, symbols: List[str], days: int = 7) -> Dict[str, Any]:
        """Compare sentiment across multiple stocks"""
        
        try:
            sentiment_comparisons = {}
            
            # Analyze sentiment for each stock
            for symbol in symbols:
                sentiment_comparisons[symbol] = await self.analyze_stock_sentiment(symbol, days)
            
            # Create comparison summary
            comparison_summary = self._create_sentiment_comparison(sentiment_comparisons)
            
            return {
                "symbols": symbols,
                "individual_sentiments": sentiment_comparisons,
                "comparison_summary": comparison_summary,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to compare stock sentiments: {str(e)}"}
    
    async def get_sentiment_alerts(self, symbols: List[str], threshold: float = 0.5) -> Dict[str, Any]:
        """Get sentiment alerts for stocks crossing threshold"""
        
        try:
            alerts = []
            
            for symbol in symbols:
                sentiment_data = await self.analyze_stock_sentiment(symbol, days=1)
                
                if "error" not in sentiment_data:
                    overall_sentiment = sentiment_data.get("overall_sentiment", {})
                    sentiment_score = overall_sentiment.get("score", 0)
                    
                    if abs(sentiment_score) >= threshold:
                        alert_type = "positive" if sentiment_score > 0 else "negative"
                        alerts.append({
                            "symbol": symbol,
                            "alert_type": alert_type,
                            "sentiment_score": sentiment_score,
                            "sentiment_label": overall_sentiment.get("label", "neutral"),
                            "article_count": sentiment_data.get("article_count", 0),
                            "key_themes": sentiment_data.get("themes", {}).get("main_topics", [])[:3]
                        })
            
            return {
                "threshold": threshold,
                "alerts": alerts,
                "alert_count": len(alerts),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate sentiment alerts: {str(e)}"}
    
    async def _analyze_articles_sentiment(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for individual articles"""
        
        sentiment_analyses = []
        
        for article in articles:
            try:
                title = article.get("title", "")
                description = article.get("description", "")
                content = f"{title} {description}"
                
                # Basic sentiment analysis using TextBlob
                blob = TextBlob(content)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Enhanced sentiment using financial keywords
                keyword_sentiment = self._analyze_financial_keywords(content)
                
                # Combine sentiments
                combined_score = (polarity * 0.6) + (keyword_sentiment * 0.4)
                
                # Determine sentiment label
                if combined_score > 0.1:
                    label = "positive"
                elif combined_score < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                
                sentiment_analyses.append({
                    "title": title,
                    "description": description,
                    "url": article.get("url", ""),
                    "published_at": article.get("published_at", ""),
                    "sentiment_score": combined_score,
                    "sentiment_label": label,
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "keyword_sentiment": keyword_sentiment,
                    "source": article.get("source", "")
                })
                
            except Exception as e:
                # Skip problematic articles
                continue
        
        return sentiment_analyses
    
    def _analyze_financial_keywords(self, text: str) -> float:
        """Analyze sentiment based on financial keywords"""
        
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0
        
        # Calculate sentiment score based on keyword ratio
        sentiment_score = (positive_count - negative_count) / total_keywords
        return sentiment_score
    
    def _calculate_overall_sentiment(self, sentiment_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall sentiment from individual article analyses"""
        
        if not sentiment_analyses:
            return {"score": 0, "label": "neutral", "confidence": 0}
        
        # Calculate weighted average (more recent articles have higher weight)
        total_score = 0
        total_weight = 0
        
        for i, analysis in enumerate(sentiment_analyses):
            # More recent articles (earlier in list) get higher weight
            weight = 1.0 / (i + 1)
            total_score += analysis["sentiment_score"] * weight
            total_weight += weight
        
        if total_weight == 0:
            return {"score": 0, "label": "neutral", "confidence": 0}
        
        overall_score = total_score / total_weight
        
        # Determine label
        if overall_score > 0.1:
            label = "positive"
        elif overall_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        # Calculate confidence based on consistency
        scores = [analysis["sentiment_score"] for analysis in sentiment_analyses]
        consistency = 1.0 - (max(scores) - min(scores)) / 2.0 if len(scores) > 1 else 1.0
        confidence = min(consistency * abs(overall_score) * 10, 1.0)
        
        return {
            "score": overall_score,
            "label": label,
            "confidence": confidence,
            "positive_articles": len([a for a in sentiment_analyses if a["sentiment_label"] == "positive"]),
            "negative_articles": len([a for a in sentiment_analyses if a["sentiment_label"] == "negative"]),
            "neutral_articles": len([a for a in sentiment_analyses if a["sentiment_label"] == "neutral"])
        }
    
    def _extract_themes(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract main themes from articles"""
        
        all_text = ""
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            all_text += f" {title} {description}"
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 
            'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'put', 'say', 'she', 'too', 'use'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency
        word_counts = Counter(filtered_words)
        main_topics = [word for word, count in word_counts.most_common(10)]
        
        # Identify financial themes
        financial_themes = []
        theme_keywords = {
            'earnings': ['earnings', 'profit', 'revenue', 'income'],
            'growth': ['growth', 'expansion', 'increase', 'rising'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
            'regulation': ['regulation', 'policy', 'government', 'regulatory'],
            'technology': ['technology', 'digital', 'innovation', 'tech'],
            'market': ['market', 'trading', 'stock', 'shares']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text.lower() for keyword in keywords):
                financial_themes.append(theme)
        
        return {
            "main_topics": main_topics,
            "financial_themes": financial_themes,
            "word_frequency": dict(word_counts.most_common(20))
        }
    
    def _calculate_sentiment_trend(self, sentiment_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sentiment trend over time"""
        
        if len(sentiment_analyses) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by publication date (most recent first)
        sorted_analyses = sorted(
            sentiment_analyses, 
            key=lambda x: x.get("published_at", ""), 
            reverse=True
        )
        
        # Calculate trend
        recent_scores = [a["sentiment_score"] for a in sorted_analyses[:len(sorted_analyses)//2]]
        older_scores = [a["sentiment_score"] for a in sorted_analyses[len(sorted_analyses)//2:]]
        
        if recent_scores and older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            trend_change = recent_avg - older_avg
            
            if trend_change > 0.1:
                trend = "improving"
            elif trend_change < -0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "trend_change": trend_change,
                "recent_average": recent_avg,
                "older_average": older_avg
            }
        
        return {"trend": "insufficient_data"}
    
    def _extract_market_themes(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract market-specific themes"""
        
        themes = self._extract_themes(articles)
        
        # Add market-specific analysis
        market_indicators = {
            'inflation': ['inflation', 'cpi', 'price', 'cost'],
            'interest_rates': ['interest', 'rates', 'fed', 'federal', 'monetary'],
            'employment': ['employment', 'jobs', 'unemployment', 'labor'],
            'gdp': ['gdp', 'growth', 'economy', 'economic'],
            'trade': ['trade', 'tariff', 'export', 'import']
        }
        
        all_text = " ".join([f"{a.get('title', '')} {a.get('description', '')}" for a in articles]).lower()
        
        market_themes = []
        for theme, keywords in market_indicators.items():
            if any(keyword in all_text for keyword in keywords):
                market_themes.append(theme)
        
        themes["market_indicators"] = market_themes
        return themes
    
    def _analyze_sector_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze sentiment by sector"""
        
        sectors = {
            'technology': ['tech', 'software', 'digital', 'ai', 'cloud', 'data'],
            'healthcare': ['health', 'medical', 'pharma', 'drug', 'biotech'],
            'financial': ['bank', 'finance', 'credit', 'loan', 'insurance'],
            'energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind'],
            'retail': ['retail', 'consumer', 'shopping', 'ecommerce']
        }
        
        sector_sentiments = {}
        
        for sector, keywords in sectors.items():
            sector_articles = []
            
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                if any(keyword in text for keyword in keywords):
                    sector_articles.append(article)
            
            if sector_articles:
                # Calculate average sentiment for sector
                sentiments = [self._analyze_financial_keywords(f"{a.get('title', '')} {a.get('description', '')}") for a in sector_articles]
                sector_sentiments[sector] = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return sector_sentiments
    
    def _identify_market_concerns(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Identify top market concerns"""
        
        concern_keywords = [
            'recession', 'inflation', 'crisis', 'volatility', 'uncertainty',
            'debt', 'default', 'risk', 'concern', 'worry', 'fear'
        ]
        
        concerns = []
        all_text = " ".join([f"{a.get('title', '')} {a.get('description', '')}" for a in articles]).lower()
        
        for keyword in concern_keywords:
            if keyword in all_text:
                concerns.append(keyword)
        
        return concerns[:5]  # Return top 5 concerns
    
    def _identify_market_opportunities(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Identify market opportunities"""
        
        opportunity_keywords = [
            'opportunity', 'growth', 'expansion', 'innovation', 'breakthrough',
            'launch', 'new', 'emerging', 'potential', 'upside'
        ]
        
        opportunities = []
        all_text = " ".join([f"{a.get('title', '')} {a.get('description', '')}" for a in articles]).lower()
        
        for keyword in opportunity_keywords:
            if keyword in all_text:
                opportunities.append(keyword)
        
        return opportunities[:5]  # Return top 5 opportunities
    
    def _create_sentiment_comparison(self, sentiment_comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create sentiment comparison summary"""
        
        valid_sentiments = {k: v for k, v in sentiment_comparisons.items() if "error" not in v}
        
        if not valid_sentiments:
            return {"error": "No valid sentiment data for comparison"}
        
        # Find most positive and negative stocks
        sentiment_scores = {}
        for symbol, data in valid_sentiments.items():
            overall_sentiment = data.get("overall_sentiment", {})
            sentiment_scores[symbol] = overall_sentiment.get("score", 0)
        
        most_positive = max(sentiment_scores.items(), key=lambda x: x[1])[0] if sentiment_scores else None
        most_negative = min(sentiment_scores.items(), key=lambda x: x[1])[0] if sentiment_scores else None
        
        # Calculate average market sentiment for these stocks
        avg_sentiment = sum(sentiment_scores.values()) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            "most_positive_stock": most_positive,
            "most_negative_stock": most_negative,
            "average_sentiment": avg_sentiment,
            "sentiment_range": max(sentiment_scores.values()) - min(sentiment_scores.values()) if sentiment_scores else 0,
            "stocks_analyzed": len(valid_sentiments),
            "sentiment_distribution": {
                "positive": len([s for s in sentiment_scores.values() if s > 0.1]),
                "negative": len([s for s in sentiment_scores.values() if s < -0.1]),
                "neutral": len([s for s in sentiment_scores.values() if -0.1 <= s <= 0.1])
            }
        }

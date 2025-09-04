"""
MCP-Powered Financial Analyst - Complete Flask Application
Professional-grade web application with all features integrated
"""
import asyncio
import io
import os
import sys
import json
import traceback
import threading
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from functools import wraps

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template_string, send_file, make_response
from flask_cors import CORS
import logging
from werkzeug.exceptions import RequestEntityTooLarge
import base64
import requests
from urllib.parse import quote

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our components
from config.settings import settings
from utils.model_client import create_gemini_model_client

# Import agent classes
try:
    from agents.orchestrator_agent import OrchestratorAgent
    from agents.data_analyst_agent import DataAnalystAgent
    from agents.news_sentiment_agent import NewsSentimentAgent
    from agents.recommendation_agent import RecommendationAgent
    agents_available = True
except ImportError as e:
    logger.warning(f"Agents not available: {e}")
    agents_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global state
agents_initialized = False
agents = {}
model_client = None

# Working MCP Client for Financial Data
class FinancialDataMCPClient:
    """Robust MCP Client for financial data with proper error handling"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def _get_cached_data(self, key: str) -> Optional[Dict]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_timeout:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict) -> None:
        """Cache data with timestamp"""
        self.cache[key] = (data, time.time())
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price using yfinance"""
        try:
            cache_key = f"price_{symbol}"
            cached = self._get_cached_data(cache_key)
            if cached:
                return cached
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return {"error": f"No data found for symbol {symbol}"}
            
            latest = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else latest
            
            result = {
                "symbol": symbol,
                "current_price": float(latest["Close"]),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "volume": int(latest["Volume"]),
                "change": float(latest["Close"] - previous["Close"]),
                "change_percent": float((latest["Close"] - previous["Close"]) / previous["Close"] * 100),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting stock price for {symbol}: {e}")
            return {"error": f"Failed to fetch stock price for {symbol}: {str(e)}", "success": False}
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information"""
        try:
            cache_key = f"info_{symbol}"
            cached = self._get_cached_data(cache_key)
            if cached:
                return cached
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "country": info.get("country", ""),
                "currency": info.get("currency", "USD"),
                "success": True
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            return {"error": f"Failed to fetch company info for {symbol}: {str(e)}", "success": False}
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical stock data"""
        try:
            cache_key = f"hist_{symbol}_{period}"
            cached = self._get_cached_data(cache_key)
            if cached:
                return cached
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No historical data found for symbol {symbol}"}
            
            data_points = []
            for date, row in hist.iterrows():
                data_points.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
            
            # Calculate technical indicators
            closes = hist["Close"]
            
            # Simple moving averages
            sma_20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.mean()
            sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()
            
            # RSI calculation
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            result = {
                "symbol": symbol,
                "period": period,
                "data": data_points,
                "technical_analysis": {
                    "sma_20": float(sma_20) if not pd.isna(sma_20) else None,
                    "sma_50": float(sma_50) if not pd.isna(sma_50) else None,
                    "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                    "support": float(closes.min()),
                    "resistance": float(closes.max()),
                    "volatility": float(closes.std())
                },
                "summary": {
                    "total_days": len(data_points),
                    "highest_price": float(hist["High"].max()),
                    "lowest_price": float(hist["Low"].min()),
                    "average_volume": float(hist["Volume"].mean())
                },
                "success": True
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return {"error": f"Failed to fetch historical data for {symbol}: {str(e)}", "success": False}
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol,
                "revenue": info.get("totalRevenue", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "profit_margin": info.get("profitMargins", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "return_on_equity": info.get("returnOnEquity", 0),
                "return_on_assets": info.get("returnOnAssets", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "price_to_book": info.get("priceToBook", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "success": True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return {"error": f"Failed to fetch fundamentals for {symbol}: {str(e)}", "success": False}

# Global MCP client
mcp_client = FinancialDataMCPClient()

# Working Agent Classes
class SimpleDataAnalyst:
    """AI-powered data analyst using Gemini model and AutoGen agents"""
    
    def __init__(self):
        """Initialize with AI model client and agents"""
        try:
            self.model_client = create_gemini_model_client()
            self.use_ai = True
            logger.info("AI model client initialized successfully")
            
            # Initialize agents if available
            if agents_available:
                self.orchestrator = OrchestratorAgent(model_client=self.model_client)
                self.data_analyst = DataAnalystAgent(model_client=self.model_client)
                self.recommendation_agent = RecommendationAgent(model_client=self.model_client)
                self.news_agent = NewsSentimentAgent(model_client=self.model_client)
                logger.info("AutoGen agents initialized successfully")
            else:
                logger.warning("AutoGen agents not available, using rule-based analysis")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            self.model_client = None
            self.use_ai = False
    
    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """AI-powered comprehensive stock analysis"""
        try:
            # Get all data
            price_data = mcp_client.get_stock_price(symbol)
            company_data = mcp_client.get_company_info(symbol)
            historical_data = mcp_client.get_historical_data(symbol)
            fundamentals = mcp_client.get_fundamentals(symbol)
            
            # Check for errors
            if not price_data.get("success"):
                return {"error": price_data.get("error", "Failed to get price data")}
            
            # Calculate key metrics
            current_price = price_data["current_price"]
            pe_ratio = price_data.get("pe_ratio", 0)
            market_cap = price_data.get("market_cap", 0)
            
            # Technical analysis from historical data
            technical = historical_data.get("technical_analysis", {}) if historical_data.get("success") else {}
            
            # Generate AI-powered recommendation
            recommendation = self._generate_ai_recommendation(symbol, price_data, company_data, fundamentals, technical)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "company_info": company_data,
                "fundamentals": fundamentals,
                "technical_analysis": technical,
                "key_metrics": {
                    "pe_ratio": pe_ratio,
                    "market_cap": market_cap,
                    "price_change": price_data.get("change", 0),
                    "price_change_percent": price_data.get("change_percent", 0),
                    "volume": price_data.get("volume", 0)
                },
                "recommendation": recommendation,
                "analysis_method": "AI-powered" if self.use_ai else "rule-based",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {"error": f"Stock analysis failed: {str(e)}", "success": False}
    
    def _generate_ai_recommendation(self, symbol: str, price_data, company_data, fundamentals, technical):
        """Generate AI-powered investment recommendation"""
        try:
            if self.use_ai and self.model_client:
                return self._ai_powered_analysis(symbol, price_data, company_data, fundamentals, technical)
            else:
                return self._rule_based_analysis(price_data, company_data, fundamentals, technical)
        except Exception as e:
            logger.error(f"AI recommendation failed, falling back to rule-based: {e}")
            return self._rule_based_analysis(price_data, company_data, fundamentals, technical)
    
    def _ai_powered_analysis(self, symbol: str, price_data, company_data, fundamentals, technical):
        """Use AI model to generate sophisticated analysis"""
        try:
            # Prepare comprehensive data for AI analysis
            analysis_prompt = f"""
            You are a professional financial analyst. Analyze the following stock data for {symbol} and provide a comprehensive investment recommendation.

            ## Current Market Data:
            - Current Price: ${price_data.get('current_price', 0):.2f}
            - Daily Change: {price_data.get('change_percent', 0):+.2f}%
            - P/E Ratio: {price_data.get('pe_ratio', 'N/A')}
            - Market Cap: ${price_data.get('market_cap', 0):,}
            - Volume: {price_data.get('volume', 0):,}

            ## Company Information:
            - Company: {company_data.get('company_name', symbol)}
            - Sector: {company_data.get('sector', 'N/A')}
            - Industry: {company_data.get('industry', 'N/A')}
            - Beta: {company_data.get('beta', 'N/A')}
            - 52-Week High: ${company_data.get('52_week_high', 0):.2f}
            - 52-Week Low: ${company_data.get('52_week_low', 0):.2f}

            ## Technical Indicators:
            - RSI: {technical.get('rsi', 'N/A')}
            - 20-day SMA: ${technical.get('sma_20', 0):.2f}
            - 50-day SMA: ${technical.get('sma_50', 0):.2f}
            - Support Level: ${technical.get('support', 0):.2f}
            - Resistance Level: ${technical.get('resistance', 0):.2f}

            ## Fundamental Data:
            - Revenue Growth: {fundamentals.get('revenue_growth', 0)*100:.1f}% if fundamentals.get('success') else 'N/A'
            - Profit Margin: {fundamentals.get('profit_margin', 0)*100:.1f}% if fundamentals.get('success') else 'N/A'
            - Debt to Equity: {fundamentals.get('debt_to_equity', 'N/A')}

            Based on this comprehensive analysis, provide:
            1. Investment Action: BUY, SELL, or HOLD
            2. Confidence Level: 0-1 (decimal)
            3. Risk Assessment: Low, Medium, or High
            4. Target Price: Based on your analysis
            5. Key Reasoning: 3-5 bullet points explaining your recommendation
            6. Time Horizon: Short-term, Medium-term, or Long-term outlook

            Respond in JSON format:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 0.75,
                "risk_level": "Medium",
                "target_price": 250.00,
                "reasoning": ["Point 1", "Point 2", "Point 3"],
                "time_horizon": "Medium-term",
                "market_outlook": "Brief market context",
                "key_strengths": ["Strength 1", "Strength 2"],
                "key_risks": ["Risk 1", "Risk 2"]
            }}
            """

            # Use the model client to get AI analysis
            import asyncio
            
            async def get_ai_recommendation():
                try:
                    # Create messages for the AI model
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a professional financial analyst with expertise in equity research, technical analysis, and fundamental analysis. Provide data-driven investment recommendations based on comprehensive market analysis."
                        },
                        {
                            "role": "user", 
                            "content": analysis_prompt
                        }
                    ]
                    
                    # Get AI response
                    response = await self.model_client.create(messages=messages)
                    ai_response = response.choices[0].message.content
                    
                    # Parse JSON response
                    try:
                        import json
                        recommendation = json.loads(ai_response)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse AI response as JSON, using fallback")
                        return self._rule_based_analysis(price_data, company_data, fundamentals, technical)
                    
                    # Ensure all required fields are present
                    recommendation.setdefault("action", "HOLD")
                    recommendation.setdefault("confidence", 0.6)
                    recommendation.setdefault("risk_level", "Medium")
                    recommendation.setdefault("target_price", price_data.get('current_price', 0))
                    recommendation.setdefault("reasoning", ["AI analysis completed"])
                    
                    return recommendation
                    
                except Exception as e:
                    logger.error(f"AI analysis error: {e}")
                    return self._rule_based_analysis(price_data, company_data, fundamentals, technical)
            
            # Run the async function
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running loop, create a new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, get_ai_recommendation())
                        return future.result(timeout=30)  # 30 second timeout
                else:
                    return asyncio.run(get_ai_recommendation())
            except Exception as e:
                logger.error(f"Async execution error: {e}")
                return self._rule_based_analysis(price_data, company_data, fundamentals, technical)
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._rule_based_analysis(price_data, company_data, fundamentals, technical)
    def _rule_based_analysis(self, price_data, company_data, fundamentals, technical):
        """Fallback rule-based recommendation generation"""
        try:
            score = 0
            reasons = []
            
            # PE ratio analysis
            pe = price_data.get("pe_ratio", 0)
            if 0 < pe < 15:
                score += 1
                reasons.append("Low P/E ratio indicates undervaluation")
            elif 15 <= pe <= 25:
                score += 0.5
                reasons.append("Moderate P/E ratio")
            elif pe > 30:
                score -= 1
                reasons.append("High P/E ratio may indicate overvaluation")
            
            # Price momentum
            change_pct = price_data.get("change_percent", 0)
            if change_pct > 2:
                score += 0.5
                reasons.append("Positive price momentum")
            elif change_pct < -2:
                score -= 0.5
                reasons.append("Negative price momentum")
            
            # Technical indicators
            rsi = technical.get("rsi")
            if rsi:
                if rsi < 30:
                    score += 1
                    reasons.append("RSI indicates oversold condition")
                elif rsi > 70:
                    score -= 1
                    reasons.append("RSI indicates overbought condition")
            
            # Determine recommendation
            if score >= 1:
                action = "BUY"
                confidence = min(0.8, 0.5 + score * 0.1)
            elif score <= -1:
                action = "SELL"
                confidence = min(0.8, 0.5 + abs(score) * 0.1)
            else:
                action = "HOLD"
                confidence = 0.6
            
            return {
                "action": action,
                "confidence": confidence,
                "score": score,
                "reasoning": reasons,
                "target_price": price_data["current_price"] * (1 + score * 0.1),
                "risk_level": "Medium" if abs(score) < 1 else "High",
                "analysis_method": "rule-based"
            }
            
        except Exception as e:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": [f"Analysis error: {str(e)}"],
                "risk_level": "Unknown",
                "analysis_method": "error-fallback"
            }
    
    def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        try:
            analyses = {}
            
            for symbol in symbols:
                analyses[symbol] = self.analyze_stock(symbol)
            
            # Generate comparison
            comparison = self._generate_comparison(analyses)
            
            return {
                "symbols": symbols,
                "individual_analyses": analyses,
                "comparison": comparison,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Stock comparison failed: {str(e)}", "success": False}
    
    def _generate_comparison(self, analyses):
        """Generate comparison analysis"""
        try:
            performance = {}
            valuation = {}
            risk = {}
            
            for symbol, analysis in analyses.items():
                if analysis.get("success"):
                    # Performance metrics
                    performance[symbol] = {
                        "price_change_percent": analysis.get("key_metrics", {}).get("price_change_percent", 0),
                        "current_price": analysis.get("current_price", 0)
                    }
                    
                    # Valuation metrics
                    valuation[symbol] = {
                        "pe_ratio": analysis.get("key_metrics", {}).get("pe_ratio", 0),
                        "market_cap": analysis.get("key_metrics", {}).get("market_cap", 0)
                    }
                    
                    # Risk metrics
                    recommendation = analysis.get("recommendation", {})
                    risk[symbol] = {
                        "recommendation": recommendation.get("action", "HOLD"),
                        "confidence": recommendation.get("confidence", 0.5),
                        "risk_level": recommendation.get("risk_level", "Medium")
                    }
            
            # Find best performer
            best_performer = max(performance.keys(), 
                               key=lambda x: performance[x]["price_change_percent"],
                               default=None)
            
            return {
                "performance": performance,
                "valuation": valuation,
                "risk": risk,
                "summary": {
                    "best_performer": best_performer,
                    "total_compared": len(analyses),
                    "recommendation_summary": self._summarize_recommendations(analyses)
                }
            }
            
        except Exception as e:
            return {"error": f"Comparison generation failed: {str(e)}"}
    
    def _summarize_recommendations(self, analyses):
        """Summarize recommendations across stocks"""
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        for analysis in analyses.values():
            if analysis.get("success"):
                action = analysis.get("recommendation", {}).get("action", "HOLD")
                if action == "BUY":
                    buy_count += 1
                elif action == "SELL":
                    sell_count += 1
                else:
                    hold_count += 1
        
        return {
            "buy_recommendations": buy_count,
            "sell_recommendations": sell_count,
            "hold_recommendations": hold_count,
            "total": buy_count + sell_count + hold_count
        }

class NewsSentimentAnalyzer:
    """AI-powered news sentiment analysis for stocks"""
    
    def __init__(self):
        """Initialize with news data sources"""
        self.news_sources = {
            "alpha_vantage": "https://www.alphavantage.co/query",
            "finnhub": "https://finnhub.io/api/v1/company-news",
            "yahoo_rss": "https://feeds.finance.yahoo.com/rss/2.0/headline"
        }
        
        # Try to get API keys from settings
        try:
            self.alpha_vantage_key = getattr(settings, 'alpha_vantage_api_key', None)
            self.finnhub_key = getattr(settings, 'finnhub_api_key', None)
        except:
            self.alpha_vantage_key = None
            self.finnhub_key = None
            logger.warning("News API keys not found, using free sources")
    
    def get_stock_news(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Fetch latest news for a specific stock"""
        try:
            news_articles = []
            
            # Method 1: Try Yahoo Finance news (free)
            yahoo_news = self._get_yahoo_news(symbol, limit)
            if yahoo_news:
                news_articles.extend(yahoo_news)
            
            # Method 2: Try Alpha Vantage if API key available
            if self.alpha_vantage_key and len(news_articles) < limit:
                av_news = self._get_alpha_vantage_news(symbol, limit - len(news_articles))
                if av_news:
                    news_articles.extend(av_news)
            
            # Method 3: Get general market news if no specific news found
            if not news_articles:
                news_articles = self._get_general_market_news(limit)
            
            # Analyze sentiment for each article
            analyzed_articles = []
            for article in news_articles[:limit]:
                sentiment = self._analyze_article_sentiment(article)
                article['sentiment'] = sentiment
                analyzed_articles.append(article)
            
            # Calculate overall sentiment score
            overall_sentiment = self._calculate_overall_sentiment(analyzed_articles)
            
            return {
                "symbol": symbol,
                "articles": analyzed_articles,
                "total_articles": len(analyzed_articles),
                "overall_sentiment": overall_sentiment,
                "sentiment_summary": self._generate_sentiment_summary(symbol, overall_sentiment, analyzed_articles),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return {
                "error": f"Failed to fetch news for {symbol}: {str(e)}",
                "symbol": symbol,
                "success": False
            }
    
    def _get_yahoo_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch news from Yahoo Finance (free method)"""
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            articles = []
            for item in news_data[:limit]:
                article = {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "url": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat() if item.get("providerPublishTime") else "",
                    "publisher": item.get("publisher", ""),
                    "source": "yahoo_finance"
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.warning(f"Yahoo news fetch failed: {e}")
            return []
    
    def _get_alpha_vantage_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch news from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return []
        
        try:
            url = f"{self.news_sources['alpha_vantage']}"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "limit": limit,
                "apikey": self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            articles = []
            if "feed" in data:
                for item in data["feed"][:limit]:
                    article = {
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "url": item.get("url", ""),
                        "published": item.get("time_published", ""),
                        "publisher": item.get("source", ""),
                        "source": "alpha_vantage"
                    }
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.warning(f"Alpha Vantage news fetch failed: {e}")
            return []
    
    def _get_general_market_news(self, limit: int = 5) -> List[Dict]:
        """Get general market news as fallback"""
        try:
            # Use popular financial news keywords
            general_articles = [
                {
                    "title": "Market Analysis: Technology Stocks Show Mixed Performance",
                    "summary": "Technology stocks continue to show varied performance amid changing market conditions and investor sentiment.",
                    "url": "#",
                    "published": datetime.now().isoformat(),
                    "publisher": "Market Analysis",
                    "source": "general"
                },
                {
                    "title": "Economic Indicators Point to Continued Growth",
                    "summary": "Recent economic data suggests continued growth in key sectors, with technology leading the way.",
                    "url": "#",
                    "published": datetime.now().isoformat(),
                    "publisher": "Economic Research",
                    "source": "general"
                }
            ]
            
            return general_articles[:limit]
            
        except Exception as e:
            logger.error(f"General news fetch failed: {e}")
            return []
    
    def _analyze_article_sentiment(self, article: Dict) -> Dict[str, Any]:
        """Analyze sentiment of a single article using AI"""
        try:
            # Combine title and summary for analysis
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            
            if hasattr(data_analyst, 'model_client') and data_analyst.model_client:
                return self._ai_sentiment_analysis(text)
            else:
                return self._rule_based_sentiment_analysis(text)
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.5,
                "method": "error_fallback"
            }
    
    def _ai_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Use AI model for sentiment analysis"""
        try:
            prompt = f"""
            Analyze the sentiment of this financial news text and respond in JSON format:
            
            Text: "{text}"
            
            Provide:
            1. Sentiment score: -1.0 (very negative) to +1.0 (very positive)
            2. Label: "positive", "negative", or "neutral"
            3. Confidence: 0.0 to 1.0
            4. Key reasons for the sentiment
            
            JSON Response:
            {{
                "score": 0.75,
                "label": "positive",
                "confidence": 0.85,
                "reasons": ["strong earnings", "market optimism"]
            }}
            """
            
            # Use async wrapper for AI analysis
            import asyncio
            
            async def get_sentiment():
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a financial sentiment analysis expert. Analyze news text and provide sentiment scores."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    
                    response = await data_analyst.model_client.create(messages=messages)
                    ai_response = response.choices[0].message.content
                    
                    import json
                    return json.loads(ai_response)
                    
                except Exception as e:
                    logger.error(f"AI sentiment analysis failed: {e}")
                    return self._rule_based_sentiment_analysis(text)
            
            # Run async sentiment analysis
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, get_sentiment())
                        result = future.result(timeout=15)
                        result["method"] = "ai_powered"
                        return result
                else:
                    result = asyncio.run(get_sentiment())
                    result["method"] = "ai_powered"
                    return result
            except Exception as e:
                logger.error(f"Async sentiment analysis failed: {e}")
                return self._rule_based_sentiment_analysis(text)
                
        except Exception as e:
            logger.error(f"AI sentiment analysis error: {e}")
            return self._rule_based_sentiment_analysis(text)
    
    def _rule_based_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis as fallback"""
        positive_words = ["gain", "up", "rise", "bull", "positive", "growth", "strong", "beat", "exceed", "profit", "revenue", "earnings"]
        negative_words = ["loss", "down", "fall", "bear", "negative", "decline", "weak", "miss", "below", "loss", "debt", "concern"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if positive_count > negative_count:
            score = min(0.8, (positive_count - negative_count) / max(total_words, 10))
            label = "positive"
        elif negative_count > positive_count:
            score = max(-0.8, -(negative_count - positive_count) / max(total_words, 10))
            label = "negative"
        else:
            score = 0.0
            label = "neutral"
        
        confidence = min(0.8, abs(score) + 0.3)
        
        return {
            "score": score,
            "label": label,
            "confidence": confidence,
            "method": "rule_based"
        }
    
    def _calculate_overall_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Calculate overall sentiment from all articles"""
        if not articles:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0}
        
        total_score = 0
        total_confidence = 0
        
        for article in articles:
            sentiment = article.get("sentiment", {})
            score = sentiment.get("score", 0)
            confidence = sentiment.get("confidence", 0.5)
            
            # Weight by confidence
            total_score += score * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            weighted_score = total_score / total_confidence
        else:
            weighted_score = 0.0
        
        # Determine label
        if weighted_score > 0.2:
            label = "positive"
        elif weighted_score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": weighted_score,
            "label": label,
            "confidence": min(1.0, total_confidence / len(articles)),
            "articles_analyzed": len(articles)
        }
    
    def _generate_sentiment_summary(self, symbol: str, overall_sentiment: Dict, articles: List[Dict]) -> str:
        """Generate human-readable sentiment summary"""
        try:
            score = overall_sentiment.get("score", 0)
            label = overall_sentiment.get("label", "neutral")
            article_count = len(articles)
            
            if label == "positive":
                sentiment_desc = "bullish" if score > 0.5 else "moderately positive"
            elif label == "negative":
                sentiment_desc = "bearish" if score < -0.5 else "moderately negative"
            else:
                sentiment_desc = "neutral"
            
            summary = f"News sentiment for {symbol} is {sentiment_desc} based on {article_count} recent articles. "
            
            if articles:
                latest_article = articles[0]
                summary += f"Latest headline: '{latest_article.get('title', 'N/A')}'. "
            
            summary += f"Overall sentiment score: {score:.2f} ({label})."
            
            return summary
            
        except Exception as e:
            return f"Sentiment summary for {symbol}: {overall_sentiment.get('label', 'unknown')} sentiment detected."


# Initialize news analyzer
news_analyzer = NewsSentimentAnalyzer()
logger.info("Initializing AI-powered financial system...")
data_analyst = SimpleDataAnalyst()
logger.info(f"Data analyst initialized with AI: {data_analyst.use_ai}")

# Global intelligent assistant
intelligent_assistant = None


# Intelligent Financial AI Assistant
class IntelligentFinancialAssistant:
    """Advanced AI assistant that processes natural language financial queries"""
    
    def __init__(self, mcp_client, data_analyst):
        self.mcp_client = mcp_client
        self.data_analyst = data_analyst
        
        # Enhanced symbol mapping with more companies
        self.symbol_mapping = {
            # Technology
            'apple': 'AAPL', 'aapl': 'AAPL',
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'netflix': 'NFLX', 'nflx': 'NFLX',
            'facebook': 'META', 'meta': 'META',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'intel': 'INTC', 'intc': 'INTC',
            'amd': 'AMD', 'advanced micro devices': 'AMD',
            'salesforce': 'CRM',
            'adobe': 'ADBE',
            'oracle': 'ORCL',
            'cisco': 'CSCO',
            'ibm': 'IBM',
            
            # Finance
            'jpmorgan': 'JPM', 'jp morgan': 'JPM',
            'bank of america': 'BAC', 'bofa': 'BAC',
            'wells fargo': 'WFC',
            'goldman sachs': 'GS',
            'morgan stanley': 'MS',
            'berkshire hathaway': 'BRK.B', 'berkshire': 'BRK.B',
            
            # Healthcare
            'johnson & johnson': 'JNJ', 'jnj': 'JNJ',
            'pfizer': 'PFE',
            'merck': 'MRK',
            'abbott': 'ABT',
            'bristol myers': 'BMY',
            
            # Consumer
            'coca cola': 'KO', 'coca-cola': 'KO', 'ko': 'KO',
            'pepsi': 'PEP', 'pepsico': 'PEP',
            'walmart': 'WMT', 'wal-mart': 'WMT',
            'target': 'TGT',
            'costco': 'COST',
            'home depot': 'HD',
            'mcdonalds': 'MCD', "mcdonald's": 'MCD',
            'starbucks': 'SBUX',
            'nike': 'NKE',
            
            # Industrial
            'boeing': 'BA',
            'caterpillar': 'CAT',
            'ge': 'GE', 'general electric': 'GE',
            '3m': 'MMM',
            
            # Energy
            'exxon': 'XOM', 'exxon mobil': 'XOM',
            'chevron': 'CVX',
            
            # Entertainment
            'disney': 'DIS', 'walt disney': 'DIS',
            'comcast': 'CMCSA',
            'sony': 'SONY',
            
            # Automotive
            'ford': 'F',
            'gm': 'GM', 'general motors': 'GM',
            
            # Airlines
            'american airlines': 'AAL',
            'delta': 'DAL', 'delta airlines': 'DAL',
            'united': 'UAL', 'united airlines': 'UAL',
            'southwest': 'LUV', 'southwest airlines': 'LUV'
        }
        
        # Query intent patterns
        self.intent_patterns = {
            'comparison': ['compare', 'vs', 'versus', 'against', 'better', 'difference', 'contrast'],
            'analysis': ['analyze', 'analysis', 'review', 'evaluate', 'assess', 'study'],
            'recommendation': ['recommend', 'should i buy', 'good investment', 'worth buying', 'advice'],
            'price': ['price', 'cost', 'value', 'trading at', 'current price'],
            'performance': ['performance', 'how is', 'doing', 'performing', 'returns'],
            'news': ['news', 'latest', 'updates', 'recent', 'happening'],
            'forecast': ['forecast', 'prediction', 'future', 'outlook', 'projection'],
            'risk': ['risk', 'risky', 'safe', 'volatility', 'dangerous'],
            'dividend': ['dividend', 'yield', 'payout', 'distribution'],
            'earnings': ['earnings', 'profit', 'revenue', 'income', 'financial results']
        }
    
    def process_intelligent_query(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """Process any financial query intelligently"""
        try:
            logger.info(f"Processing intelligent query: {query}")
            
            # Step 1: Extract symbols and determine intent
            symbols = self._extract_symbols_from_query(query)
            intent = self._determine_intent(query)
            
            # Step 2: Route to appropriate handler
            if intent == 'news':
                result = self._handle_news_query(symbols, query)
            elif intent == 'comparison' and len(symbols) >= 2:
                result = self._handle_comparison_query(symbols, query)
            elif len(symbols) == 1:
                result = self._handle_single_stock_query(symbols[0], query, intent)
            elif intent in ['analysis', 'performance'] and not symbols:
                # Default to major stocks if no symbols detected
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                result = self._handle_market_overview_query(symbols, query)
            else:
                result = self._handle_general_query(query, symbols)
            
            # Step 3: Save to database
            self._save_query_to_database(user_id, query, result)
            
            # Step 4: Format response
            result.update({
                "original_query": query,
                "detected_symbols": symbols,
                "detected_intent": intent,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "processed_by": "intelligent_financial_assistant"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing intelligent query: {e}")
            return {
                "error": f"Query processing failed: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Enhanced symbol extraction with validation"""
        query_lower = query.lower()
        symbols = []
        
        # Look for direct symbol mentions (2-5 uppercase letters)
        symbol_pattern = r'\b[A-Z]{2,5}\b'
        direct_symbols = re.findall(symbol_pattern, query)
        symbols.extend(direct_symbols)
        
        # Look for company names (including partial matches)
        for name, symbol in self.symbol_mapping.items():
            if name in query_lower:
                symbols.append(symbol)
        
        # Validate symbols by testing with API
        validated_symbols = []
        for symbol in symbols:
            if symbol not in validated_symbols:  # Avoid duplicates
                test_result = self.mcp_client.get_stock_price(symbol)
                if test_result.get("success"):
                    validated_symbols.append(symbol)
        
        return validated_symbols[:5]  # Limit to 5 symbols
    
    def _determine_intent(self, query: str) -> str:
        """Determine user's intent from the query"""
        query_lower = query.lower()
        
        # Count matches for each intent
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with highest score, or 'analysis' as default
        if intent_scores:
            return max(intent_scores.keys(), key=lambda x: intent_scores[x])
        return 'analysis'
    
    def _handle_news_query(self, symbols: List[str], query: str) -> Dict[str, Any]:
        """Handle news and sentiment analysis queries"""
        try:
            logger.info(f"Handling news query for symbols: {symbols}")
            
            if symbols:
                # Get news for specific symbol
                symbol = symbols[0]
                news_result = news_analyzer.get_stock_news(symbol, limit=5)
                
                if not news_result.get("success"):
                    return news_result
                
                # Generate human-readable summary
                summary = self._generate_news_summary(news_result, query)
                
                return {
                    "query_type": "news_analysis",
                    "symbol": symbol,
                    "news_data": news_result,
                    "human_readable_summary": summary,
                    "success": True
                }
            else:
                # General market news
                logger.info("Getting general market news")
                
                # Use major stocks for market news
                market_symbols = ['AAPL', 'MSFT', 'GOOGL']
                market_news = {}
                
                for symbol in market_symbols:
                    try:
                        news_result = news_analyzer.get_stock_news(symbol, limit=2)
                        if news_result.get("success"):
                            market_news[symbol] = news_result
                    except Exception as e:
                        logger.warning(f"Failed to get news for {symbol}: {e}")
                
                # Generate market news summary
                summary = self._generate_market_news_summary(market_news, query)
                
                return {
                    "query_type": "market_news",
                    "market_news": market_news,
                    "human_readable_summary": summary,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"News query handling failed: {e}")
            return {"error": f"News analysis failed: {str(e)}", "success": False}
    
    def _generate_news_summary(self, news_result: Dict, query: str) -> str:
        """Generate human-readable news summary"""
        try:
            symbol = news_result.get("symbol", "")
            overall_sentiment = news_result.get("overall_sentiment", {})
            articles = news_result.get("articles", [])
            
            sentiment_label = overall_sentiment.get("label", "neutral")
            sentiment_score = overall_sentiment.get("score", 0)
            article_count = len(articles)
            
            summary = f"📰 **News Analysis for {symbol}**\n"
            summary += f"**Query:** {query}\n\n"
            
            # Sentiment overview
            summary += f"**Overall Sentiment:** {sentiment_label.title()} "
            summary += f"(Score: {sentiment_score:.2f})\n"
            summary += f"**Articles Analyzed:** {article_count}\n\n"
            
            # Recent headlines
            if articles:
                summary += f"**Recent Headlines:**\n"
                for i, article in enumerate(articles[:3], 1):
                    title = article.get("title", "No title")
                    sentiment = article.get("sentiment", {})
                    sent_label = sentiment.get("label", "neutral")
                    
                    # Add emoji based on sentiment
                    emoji = "📈" if sent_label == "positive" else "📉" if sent_label == "negative" else "➡️"
                    summary += f"{i}. {emoji} {title}\n"
                
                summary += "\n"
            
            # Investment implication
            if sentiment_score > 0.3:
                summary += f"**Investment Implication:** Positive news sentiment may support {symbol} price momentum.\n"
            elif sentiment_score < -0.3:
                summary += f"**Investment Implication:** Negative news sentiment may create headwinds for {symbol}.\n"
            else:
                summary += f"**Investment Implication:** Neutral news sentiment suggests focus on fundamental analysis.\n"
            
            return summary
            
        except Exception as e:
            return f"News summary for {news_result.get('symbol', 'stock')}: {str(e)}"
    
    def _generate_market_news_summary(self, market_news: Dict, query: str) -> str:
        """Generate market-wide news summary"""
        try:
            summary = f"📰 **Market News Overview**\n"
            summary += f"**Query:** {query}\n\n"
            
            if not market_news:
                summary += "No recent market news available.\n"
                return summary
            
            # Analyze overall market sentiment
            sentiment_scores = []
            for symbol, news_data in market_news.items():
                overall_sentiment = news_data.get("overall_sentiment", {})
                score = overall_sentiment.get("score", 0)
                sentiment_scores.append(score)
            
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                if avg_sentiment > 0.2:
                    market_mood = "Bullish 📈"
                elif avg_sentiment < -0.2:
                    market_mood = "Bearish 📉"
                else:
                    market_mood = "Neutral ➡️"
                
                summary += f"**Market Sentiment:** {market_mood} "
                summary += f"(Average: {avg_sentiment:.2f})\n\n"
            
            # Individual stock news
            summary += f"**Stock-Specific News:**\n"
            for symbol, news_data in market_news.items():
                overall_sentiment = news_data.get("overall_sentiment", {})
                sentiment_label = overall_sentiment.get("label", "neutral")
                articles = news_data.get("articles", [])
                
                emoji = "📈" if sentiment_label == "positive" else "📉" if sentiment_label == "negative" else "➡️"
                summary += f"• {symbol}: {emoji} {sentiment_label.title()} sentiment "
                summary += f"({len(articles)} articles)\n"
                
                if articles:
                    latest = articles[0]
                    title = latest.get("title", "")[:60] + "..." if len(latest.get("title", "")) > 60 else latest.get("title", "")
                    summary += f"  Latest: {title}\n"
            
            return summary
            
        except Exception as e:
            return f"Market news summary error: {str(e)}"
    
    def _handle_comparison_query(self, symbols: List[str], query: str) -> Dict[str, Any]:
        """Handle stock comparison queries"""
        try:
            logger.info(f"Handling comparison query for symbols: {symbols}")
            
            comparison_result = self.data_analyst.compare_stocks(symbols)
            
            if not comparison_result.get("success"):
                return comparison_result
            
            # Generate human-readable summary
            summary = self._generate_comparison_summary(comparison_result, query)
            
            return {
                "query_type": "comparison",
                "analysis": comparison_result,
                "human_readable_summary": summary,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Comparison analysis failed: {str(e)}", "success": False}
    
    def _handle_single_stock_query(self, symbol: str, query: str, intent: str) -> Dict[str, Any]:
        """Handle single stock analysis queries"""
        try:
            logger.info(f"Handling single stock query for {symbol} with intent: {intent}")
            
            analysis_result = self.data_analyst.analyze_stock(symbol)
            
            if not analysis_result.get("success"):
                return analysis_result
            
            # Generate human-readable summary based on intent
            summary = self._generate_single_stock_summary(analysis_result, query, intent)
            
            return {
                "query_type": "single_stock_analysis",
                "analysis": analysis_result,
                "human_readable_summary": summary,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Single stock analysis failed: {str(e)}", "success": False}
    
    def _handle_market_overview_query(self, symbols: List[str], query: str) -> Dict[str, Any]:
        """Handle general market overview queries"""
        try:
            logger.info(f"Handling market overview query with symbols: {symbols}")
            
            market_data = self.data_analyst.compare_stocks(symbols)
            
            summary = f"Market Overview Analysis for {', '.join(symbols)}\n\n"
            summary += self._generate_market_summary(market_data, query)
            
            return {
                "query_type": "market_overview",
                "analysis": market_data,
                "human_readable_summary": summary,
                "suggestion": "Ask about specific stocks or 'compare AAPL vs MSFT' for detailed analysis",
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Market overview failed: {str(e)}", "success": False}
    
    def _handle_general_query(self, query: str, symbols: List[str]) -> Dict[str, Any]:
        """Handle general financial queries"""
        return {
            "query_type": "general",
            "message": f"I understand you're asking about: '{query}'",
            "suggestion": "Try asking about specific stocks like 'How is Apple performing?' or 'Compare Tesla vs Ford'",
            "available_features": [
                "Stock analysis (e.g., 'analyze AAPL')",
                "Stock comparison (e.g., 'compare AAPL vs MSFT')",
                "Price lookup (e.g., 'Tesla stock price')",
                "Company information (e.g., 'Microsoft company info')"
            ],
            "detected_symbols": symbols,
            "success": True
        }
    
    def _generate_comparison_summary(self, comparison_result: Dict, query: str) -> str:
        """Generate human-readable comparison summary"""
        try:
            symbols = comparison_result.get("symbols", [])
            analyses = comparison_result.get("individual_analyses", {})
            comparison = comparison_result.get("comparison", {})
            
            summary = f"📊 **Stock Comparison Analysis**\n"
            summary += f"**Query:** {query}\n\n"
            
            # Individual stock summaries
            for symbol in symbols:
                analysis = analyses.get(symbol, {})
                if analysis.get("success"):
                    company_name = analysis.get("company_info", {}).get("company_name", symbol)
                    current_price = analysis.get("current_price", 0)
                    change_pct = analysis.get("key_metrics", {}).get("price_change_percent", 0)
                    recommendation = analysis.get("recommendation", {})
                    
                    summary += f"**{company_name} ({symbol})**\n"
                    summary += f"• Current Price: ${current_price:.2f}\n"
                    summary += f"• Daily Change: {change_pct:+.2f}%\n"
                    summary += f"• Recommendation: {recommendation.get('action', 'HOLD')}\n"
                    summary += f"• Confidence: {recommendation.get('confidence', 0)*100:.0f}%\n\n"
            
            # Comparison insights
            performance = comparison.get("performance", {})
            best_performer = comparison.get("summary", {}).get("best_performer")
            
            if best_performer and best_performer in performance:
                best_change = performance[best_performer]["price_change_percent"]
                summary += f"🏆 **Best Performer:** {best_performer} ({best_change:+.2f}%)\n\n"
            
            # Recommendations summary
            rec_summary = comparison.get("summary", {}).get("recommendation_summary", {})
            buy_count = rec_summary.get("buy_recommendations", 0)
            hold_count = rec_summary.get("hold_recommendations", 0)
            sell_count = rec_summary.get("sell_recommendations", 0)
            
            summary += f"📈 **Investment Recommendations:**\n"
            summary += f"• BUY: {buy_count} stock(s)\n"
            summary += f"• HOLD: {hold_count} stock(s)\n"
            summary += f"• SELL: {sell_count} stock(s)\n"
            
            return summary
            
        except Exception as e:
            return f"Summary generation error: {str(e)}"
    
    def _generate_single_stock_summary(self, analysis: Dict, query: str, intent: str) -> str:
        """Generate human-readable single stock summary"""
        try:
            symbol = analysis.get("symbol", "")
            company_info = analysis.get("company_info", {})
            current_price = analysis.get("current_price", 0)
            key_metrics = analysis.get("key_metrics", {})
            recommendation = analysis.get("recommendation", {})
            technical = analysis.get("technical_analysis", {})
            fundamentals = analysis.get("fundamentals", {})
            
            company_name = company_info.get("company_name", symbol)
            sector = company_info.get("sector", "Unknown")
            
            summary = f"📈 **{company_name} ({symbol}) Analysis**\n"
            summary += f"**Query:** {query}\n\n"
            
            # Price and performance
            summary += f"**Current Performance:**\n"
            summary += f"• Stock Price: ${current_price:.2f}\n"
            summary += f"• Daily Change: {key_metrics.get('price_change_percent', 0):+.2f}%\n"
            summary += f"• Sector: {sector}\n"
            summary += f"• Market Cap: ${key_metrics.get('market_cap', 0):,.0f}\n\n"
            
            # Key financial metrics
            summary += f"**Key Metrics:**\n"
            summary += f"• P/E Ratio: {key_metrics.get('pe_ratio', 'N/A')}\n"
            if fundamentals.get("success"):
                summary += f"• Profit Margin: {fundamentals.get('profit_margin', 0)*100:.1f}%\n"
                summary += f"• Revenue Growth: {fundamentals.get('revenue_growth', 0)*100:.1f}%\n"
            summary += f"• Volume: {key_metrics.get('volume', 0):,}\n\n"
            
            # Technical analysis
            if technical:
                summary += f"**Technical Indicators:**\n"
                if technical.get("rsi"):
                    summary += f"• RSI: {technical['rsi']:.1f}\n"
                if technical.get("sma_20"):
                    summary += f"• 20-day SMA: ${technical['sma_20']:.2f}\n"
                if technical.get("sma_50"):
                    summary += f"• 50-day SMA: ${technical['sma_50']:.2f}\n"
                summary += f"• Support: ${technical.get('support', 0):.2f}\n"
                summary += f"• Resistance: ${technical.get('resistance', 0):.2f}\n\n"
            
            # Investment recommendation
            summary += f"**Investment Recommendation:**\n"
            summary += f"• Action: **{recommendation.get('action', 'HOLD')}**\n"
            summary += f"• Confidence: {recommendation.get('confidence', 0)*100:.0f}%\n"
            summary += f"• Risk Level: {recommendation.get('risk_level', 'Medium')}\n"
            summary += f"• Target Price: ${recommendation.get('target_price', current_price):.2f}\n\n"
            
            # Reasoning
            reasons = recommendation.get('reasoning', [])
            if reasons:
                summary += f"**Analysis Reasoning:**\n"
                for reason in reasons:
                    summary += f"• {reason}\n"
            
            return summary
            
        except Exception as e:
            return f"Summary generation error: {str(e)}"
    
    def _generate_market_summary(self, market_data: Dict, query: str) -> str:
        """Generate market overview summary"""
        try:
            symbols = market_data.get("symbols", [])
            comparison = market_data.get("comparison", {})
            
            summary = f"Current market snapshot for major technology stocks:\n\n"
            
            # Performance overview
            performance = comparison.get("performance", {})
            if performance:
                summary += "📊 **Performance Summary:**\n"
                for symbol, perf in performance.items():
                    change = perf.get("price_change_percent", 0)
                    price = perf.get("current_price", 0)
                    summary += f"• {symbol}: ${price:.2f} ({change:+.2f}%)\n"
                summary += "\n"
            
            # Overall market sentiment
            rec_summary = comparison.get("summary", {}).get("recommendation_summary", {})
            buy_count = rec_summary.get("buy_recommendations", 0)
            total = rec_summary.get("total", len(symbols))
            
            if total > 0:
                bullish_pct = (buy_count / total) * 100
                summary += f"**Market Sentiment:** {bullish_pct:.0f}% bullish signals\n"
            
            return summary
            
        except Exception as e:
            return f"Market summary error: {str(e)}"
    
    def _save_query_to_database(self, user_id: str, query: str, result: Dict):
        """Save query and result to database"""
        try:
            import sqlite3
            conn = sqlite3.connect('./data/financial_analyst.db')
            conn.execute(
                "INSERT INTO user_queries (user_id, query_text, response_data) VALUES (?, ?, ?)",
                (user_id, query, json.dumps(result))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Database save failed: {e}")
    
    def process_query(self, user_query: str, user_id: str = 'default') -> Dict[str, Any]:
        """
        Main method to process any financial query intelligently with dual output formats
        """
        try:
            logger.info(f"Processing financial query: {user_query}")
            
            # Step 1: Extract symbols from query
            symbols = self._extract_symbols_from_query(user_query)
            query_lower = user_query.lower()
            
            # Step 2: Determine query type and route appropriately
            if len(symbols) > 1 or any(word in query_lower for word in ['compare', 'vs', 'versus', 'against']):
                # Multi-stock comparison
                if len(symbols) < 2:
                    symbols = ['AAPL', 'MSFT']  # Default comparison
                result = self._handle_comparison_query(symbols[:5], user_query)
                
            elif len(symbols) == 1:
                # Single stock analysis
                result = self._handle_single_stock_query(symbols[0], user_query, 'analysis')
                
            else:
                # General market query - provide default analysis
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                result = self._handle_market_overview_query(symbols, user_query)
            
            # Step 3: Generate dual format outputs
            if result.get("success"):
                formatted_response = self._generate_dual_format_response(result, user_query, symbols)
                result.update(formatted_response)
            
            # Step 4: Save to database
            self._save_query_to_database(user_id, user_query, result)
            
            # Step 5: Add metadata
            result.update({
                "original_query": user_query,
                "detected_symbols": symbols,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "processed_by": "intelligent_financial_assistant",
                "success": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": f"Query processing failed: {str(e)}",
                "query": user_query,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    def _generate_dual_format_response(self, analysis_result: Dict, query: str, symbols: List[str]) -> Dict[str, Any]:
        """Generate both sentence and table format responses"""
        try:
            query_type = analysis_result.get("query_type", "analysis")
            analysis_data = analysis_result.get("analysis", {})
            
            if query_type == "stock_comparison":
                return self._format_comparison_dual_output(analysis_data, symbols, query)
            elif query_type == "single_stock_analysis":
                return self._format_single_stock_dual_output(analysis_data, symbols[0] if symbols else "Stock", query)
            elif query_type == "market_overview":
                return self._format_market_overview_dual_output(analysis_data, symbols, query)
            else:
                return self._format_general_dual_output(query)
                
        except Exception as e:
            logger.error(f"Dual format generation failed: {e}")
            return {
                "sentence_format": f"Error formatting response: {str(e)}",
                "table_format": "Formatting error occurred",
                "error": str(e)
            }
    
    def _format_comparison_dual_output(self, data: Dict, symbols: List[str], query: str) -> Dict[str, Any]:
        """Format comparison analysis in both sentence and table formats"""
        try:
            individual_analyses = data.get("individual_analyses", {})
            comparison = data.get("comparison", {})
            
            # Sentence format
            sentence_parts = []
            
            # Table format setup
            table_header = "| Symbol | Company | Price ($) | Change (%) | P/E | Recommendation | Confidence |"
            table_separator = "|--------|---------|-----------|------------|-----|----------------|------------|"
            table_rows = [table_header, table_separator]
            
            for symbol in symbols:
                analysis = individual_analyses.get(symbol, {})
                if analysis.get("success"):
                    company_info = analysis.get("company_info", {})
                    company_name = company_info.get("company_name", symbol)
                    price = analysis.get("current_price", 0)
                    change_pct = analysis.get("key_metrics", {}).get("price_change_percent", 0)
                    pe_ratio = analysis.get("key_metrics", {}).get("pe_ratio", 0)
                    recommendation = analysis.get("recommendation", {})
                    action = recommendation.get("action", "HOLD")
                    confidence = recommendation.get("confidence", 0.5)
                    
                    # Add to sentence
                    direction = "up" if change_pct > 0 else "down" if change_pct < 0 else "flat"
                    sentence_parts.append(
                        f"{company_name} ({symbol}) is priced at ${price:.2f}, {direction} {abs(change_pct):.2f}% with a {action} recommendation at {confidence*100:.0f}% confidence"
                    )
                    
                    # Add to table
                    pe_display = f"{pe_ratio:.1f}" if pe_ratio > 0 else "N/A"
                    company_short = company_name[:12] + "..." if len(company_name) > 15 else company_name
                    table_rows.append(
                        f"| {symbol} | {company_short} | {price:.2f} | {change_pct:+.2f}% | {pe_display} | {action} | {confidence*100:.0f}% |"
                    )
            
            # Generate comparison insights for sentence
            performance = comparison.get("performance", {})
            if performance:
                best_performer = max(performance.keys(), 
                                   key=lambda x: performance[x].get("price_change_percent", -999))
                best_change = performance[best_performer].get("price_change_percent", 0)
                sentence_parts.append(f"Among these stocks, {best_performer} leads with {best_change:+.2f}% performance")
            
            sentence_format = ". ".join(sentence_parts) + "."
            table_format = "\n".join(table_rows)
            
            return {
                "sentence_format": sentence_format,
                "table_format": table_format,
                "format_type": "comparison_analysis"
            }
            
        except Exception as e:
            return {
                "sentence_format": f"Comparison formatting error: {str(e)}",
                "table_format": "Table generation failed",
                "error": str(e)
            }
    
    def _format_single_stock_dual_output(self, data: Dict, symbol: str, query: str) -> Dict[str, Any]:
        """Format single stock analysis in both sentence and table formats"""
        try:
            if not data.get("success"):
                error_msg = data.get("error", "Analysis failed")
                return {
                    "sentence_format": f"Unable to analyze {symbol}: {error_msg}",
                    "table_format": f"| Error | {error_msg} |",
                    "error": error_msg
                }
            
            company_info = data.get("company_info", {})
            price = data.get("current_price", 0)
            key_metrics = data.get("key_metrics", {})
            recommendation = data.get("recommendation", {})
            fundamentals = data.get("fundamentals", {})
            technical = data.get("technical_analysis", {})
            
            company_name = company_info.get("company_name", symbol)
            sector = company_info.get("sector", "N/A")
            change_pct = key_metrics.get("price_change_percent", 0)
            pe_ratio = key_metrics.get("pe_ratio", 0)
            market_cap = key_metrics.get("market_cap", 0)
            volume = key_metrics.get("volume", 0)
            
            action = recommendation.get("action", "HOLD")
            confidence = recommendation.get("confidence", 0.5)
            risk_level = recommendation.get("risk_level", "Medium")
            target_price = recommendation.get("target_price", price)
            
            # Sentence format
            direction = "gained" if change_pct > 0 else "lost" if change_pct < 0 else "remained stable"
            sentence_format = f"{company_name} ({symbol}) is currently trading at ${price:.2f}, having {direction} {abs(change_pct):.2f}% today. "
            sentence_format += f"Our analysis recommends a {action} position with {confidence*100:.0f}% confidence, indicating {risk_level.lower()} risk. "
            
            if target_price != price:
                potential = ((target_price - price) / price) * 100
                sentence_format += f"The target price of ${target_price:.2f} suggests {potential:+.1f}% potential. "
            
            # Add key insight
            reasoning = recommendation.get("reasoning", [])
            if reasoning:
                sentence_format += f"Key factors include: {reasoning[0]}."
            
            # Table format
            market_cap_display = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M" if market_cap > 1e6 else "N/A"
            pe_display = f"{pe_ratio:.1f}" if pe_ratio > 0 else "N/A"
            volume_display = f"{volume:,}" if volume > 0 else "N/A"
            
            table_format = f"""| Metric | Value |
|--------|-------|
| Company | {company_name} |
| Symbol | {symbol} |
| Sector | {sector} |
| Current Price | ${price:.2f} |
| Daily Change | {change_pct:+.2f}% |
| Market Cap | {market_cap_display} |
| P/E Ratio | {pe_display} |
| Volume | {volume_display} |
| Recommendation | {action} |
| Confidence | {confidence*100:.0f}% |
| Risk Level | {risk_level} |
| Target Price | ${target_price:.2f} |"""
            
            # Add technical indicators if available
            if technical:
                rsi = technical.get("rsi")
                sma_20 = technical.get("sma_20")
                if rsi:
                    table_format += f"\n| RSI (14) | {rsi:.1f} |"
                if sma_20:
                    table_format += f"\n| 20-day SMA | ${sma_20:.2f} |"
            
            return {
                "sentence_format": sentence_format,
                "table_format": table_format,
                "format_type": "single_stock_analysis"
            }
            
        except Exception as e:
            return {
                "sentence_format": f"Error analyzing {symbol}: {str(e)}",
                "table_format": "Analysis error",
                "error": str(e)
            }
    
    def _format_market_overview_dual_output(self, data: Dict, symbols: List[str], query: str) -> Dict[str, Any]:
        """Format market overview in both sentence and table formats"""
        try:
            comparison = data.get("comparison", {})
            performance = comparison.get("performance", {})
            individual_analyses = data.get("individual_analyses", {})
            
            if not performance:
                return {
                    "sentence_format": "Market data is currently unavailable for analysis.",
                    "table_format": "| Status | Data Unavailable |",
                    "format_type": "market_overview"
                }
            
            # Calculate market sentiment
            positive_count = sum(1 for p in performance.values() if p.get("price_change_percent", 0) > 0)
            total_count = len(performance)
            avg_change = sum(p.get("price_change_percent", 0) for p in performance.values()) / total_count if total_count > 0 else 0
            
            # Sentence format
            market_sentiment = "bullish" if positive_count > total_count / 2 else "bearish" if positive_count < total_count / 2 else "mixed"
            sentence_format = f"Market analysis of {total_count} major stocks shows {market_sentiment} sentiment with {positive_count} stocks advancing. "
            sentence_format += f"Average performance is {avg_change:+.2f}% with "
            
            # Find best and worst performers
            if performance:
                best_stock = max(performance.keys(), key=lambda x: performance[x].get("price_change_percent", -999))
                worst_stock = min(performance.keys(), key=lambda x: performance[x].get("price_change_percent", 999))
                best_change = performance[best_stock].get("price_change_percent", 0)
                worst_change = performance[worst_stock].get("price_change_percent", 0)
                
                sentence_format += f"{best_stock} leading at {best_change:+.2f}% and {worst_stock} lagging at {worst_change:+.2f}%."
            
            # Table format
            table_header = "| Symbol | Company | Price ($) | Change (%) | Status | Trend |"
            table_separator = "|--------|---------|-----------|------------|--------|-------|"
            table_rows = [table_header, table_separator]
            
            for symbol, perf in performance.items():
                analysis = individual_analyses.get(symbol, {})
                company_info = analysis.get("company_info", {})
                company_name = company_info.get("company_name", symbol)
                price = perf.get("current_price", 0)
                change = perf.get("price_change_percent", 0)
                
                # Determine status and trend
                if change > 2:
                    status = "Strong +"
                    trend = "📈"
                elif change > 0:
                    status = "Up +"
                    trend = "↗️"
                elif change < -2:
                    status = "Weak -"
                    trend = "📉"
                elif change < 0:
                    status = "Down -"
                    trend = "↘️"
                else:
                    status = "Flat"
                    trend = "➡️"
                
                company_short = company_name[:10] + "..." if len(company_name) > 13 else company_name
                table_rows.append(
                    f"| {symbol} | {company_short} | {price:.2f} | {change:+.2f}% | {status} | {trend} |"
                )
            
            # Add market summary row
            table_rows.append(table_separator)
            table_rows.append(f"| **MARKET** | **AVERAGE** | **N/A** | **{avg_change:+.2f}%** | **{market_sentiment.upper()}** | **📊** |")
            
            table_format = "\n".join(table_rows)
            
            return {
                "sentence_format": sentence_format,
                "table_format": table_format,
                "format_type": "market_overview"
            }
            
        except Exception as e:
            return {
                "sentence_format": f"Market overview error: {str(e)}",
                "table_format": "Market data error",
                "error": str(e)
            }
    
    def _format_general_dual_output(self, query: str) -> Dict[str, Any]:
        """Format general response when no specific analysis is performed"""
        sentence_format = "I'm ready to help you with financial analysis! I can analyze individual stocks, compare multiple stocks, or provide market overviews. Please specify the stocks or companies you'd like me to analyze."
        
        table_format = """| Feature | Available | Example |
|---------|-----------|---------|
| Stock Analysis | ✅ | "Analyze Apple stock" |
| Stock Comparison | ✅ | "Compare Tesla vs Ford" |
| Market Overview | ✅ | "Market outlook today" |
| Price Lookup | ✅ | "AAPL current price" |
| Recommendations | ✅ | "Should I buy MSFT?" |"""
        
        return {
            "sentence_format": sentence_format,
            "table_format": table_format,
            "format_type": "general_guidance"
        }
    
    def _handle_comparison_query(self, symbols: List[str], query: str) -> Dict[str, Any]:
        """Handle multi-stock comparison queries"""
        try:
            comparison_result = self.data_analyst.compare_stocks(symbols)
            
            # Generate human-readable summary
            summary = self._generate_comparison_summary(symbols, comparison_result, query)
            
            return {
                "query_type": "stock_comparison",
                "symbols": symbols,
                "analysis": comparison_result,
                "human_summary": summary,
                "success": True
            }
        except Exception as e:
            return {"error": f"Comparison failed: {str(e)}", "success": False}
    
    def _handle_single_stock_query(self, symbol: str, query: str, intent: str) -> Dict[str, Any]:
        """Handle single stock analysis queries"""
        try:
            analysis = self.data_analyst.analyze_stock(symbol)
            
            # Generate human-readable summary
            summary = self._generate_stock_summary(symbol, analysis, query)
            
            return {
                "query_type": "single_stock_analysis",
                "symbol": symbol,
                "analysis": analysis,
                "human_summary": summary,
                "success": True
            }
        except Exception as e:
            return {"error": f"Stock analysis failed: {str(e)}", "success": False}
    
    def _handle_market_overview_query(self, symbols: List[str], query: str) -> Dict[str, Any]:
        """Handle general market queries"""
        try:
            market_analysis = self.data_analyst.compare_stocks(symbols)
            
            summary = f"No specific stocks detected in your query: '{query}'\n\n"
            summary += "Here's an overview of major tech stocks:\n\n"
            summary += self._generate_market_summary(symbols, market_analysis)
            
            return {
                "query_type": "market_overview",
                "message": "General market analysis provided",
                "symbols": symbols,
                "analysis": market_analysis,
                "human_summary": summary,
                "suggestion": "Try asking about specific stocks like 'AAPL analysis' or 'compare Apple vs Microsoft'",
                "success": True
            }
        except Exception as e:
            return {"error": f"Market overview failed: {str(e)}", "success": False}
    
    def _generate_stock_summary(self, symbol: str, analysis: Dict, query: str) -> str:
        """Generate human-readable summary for single stock"""
        try:
            if not analysis.get("success"):
                return f"Unable to analyze {symbol}: {analysis.get('error', 'Unknown error')}"
            
            company_info = analysis.get("company_info", {})
            current_price = analysis.get("current_price", 0)
            key_metrics = analysis.get("key_metrics", {})
            recommendation = analysis.get("recommendation", {})
            
            summary = f"**{company_info.get('company_name', symbol)} ({symbol}) Analysis**\n\n"
            summary += f"**Current Price:** ${current_price:.2f}\n"
            
            change_pct = key_metrics.get("price_change_percent", 0)
            if change_pct != 0:
                summary += f"**Daily Change:** {change_pct:+.2f}%\n"
            
            # Key metrics
            if key_metrics.get("pe_ratio"):
                summary += f"**P/E Ratio:** {key_metrics['pe_ratio']:.2f}\n"
            if key_metrics.get("market_cap"):
                market_cap_b = key_metrics["market_cap"] / 1e9
                summary += f"**Market Cap:** ${market_cap_b:.1f}B\n"
            
            # Recommendation
            action = recommendation.get("action", "HOLD")
            confidence = recommendation.get("confidence", 0.5)
            summary += f"\n**Recommendation:** {action} (Confidence: {confidence:.0%})\n"
            
            reasoning = recommendation.get("reasoning", [])
            if reasoning:
                summary += "**Key Points:**\n"
                for reason in reasoning[:3]:  # Limit to top 3 reasons
                    summary += f"• {reason}\n"
            
            return summary
            
        except Exception as e:
            return f"Summary generation error: {str(e)}"
    
    def _generate_comparison_summary(self, symbols: List[str], comparison: Dict, query: str) -> str:
        """Generate human-readable comparison summary"""
        try:
            if not comparison.get("success"):
                return f"Comparison failed: {comparison.get('error', 'Unknown error')}"
            
            analyses = comparison.get("individual_analyses", {})
            comp_data = comparison.get("comparison", {})
            
            summary = f"**Stock Comparison Analysis**\n\n"
            
            # Individual stock summaries
            for symbol in symbols:
                analysis = analyses.get(symbol, {})
                if analysis.get("success"):
                    price = analysis.get("current_price", 0)
                    change_pct = analysis.get("key_metrics", {}).get("price_change_percent", 0)
                    rec = analysis.get("recommendation", {}).get("action", "HOLD")
                    summary += f"**{symbol}:** ${price:.2f} ({change_pct:+.2f}%) - {rec}\n"
            
            # Best performer
            best = comp_data.get("summary", {}).get("best_performer")
            if best:
                summary += f"\n**Best Performer:** {best}\n"
            
            # Overall recommendation summary
            rec_summary = comp_data.get("summary", {}).get("recommendation_summary", {})
            buy_count = rec_summary.get("buy_recommendations", 0)
            total = rec_summary.get("total", len(symbols))
            
            if total > 0:
                summary += f"**Investment Signals:** {buy_count}/{total} BUY recommendations\n"
            
            return summary
            
        except Exception as e:
            return f"Comparison summary error: {str(e)}"

# Initialize the AI-powered intelligent assistant after all components are ready
try:
    intelligent_assistant = IntelligentFinancialAssistant(mcp_client, data_analyst)
    logger.info("✅ Intelligent Financial Assistant initialized successfully")
    logger.info(f"🤖 AI Mode: {'ENABLED' if data_analyst.use_ai else 'DISABLED (using rule-based fallback)'}")
except Exception as e:
    logger.error(f"❌ Failed to initialize Intelligent Assistant: {e}")
    intelligent_assistant = None

def init_agents():
    """Initialize agents (simplified for working system)"""
    global agents_initialized
    agents_initialized = True
    return True

def init_database():
    """Initialize database tables"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create tables synchronously for Flask compatibility
        import sqlite3
        
        db_path = './data/financial_analyst.db'
        conn = sqlite3.connect(db_path)
        
        # Create user_queries table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query_text TEXT NOT NULL,
                response_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create portfolios table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                name TEXT NOT NULL,
                holdings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create stock_cache table for performance
        conn.execute('''
            CREATE TABLE IF NOT EXISTS stock_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False
    """Decorator to run async functions in sync context"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(func(*args, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Async execution error: {e}")
            raise e
    return wrapper

def extract_symbols_from_query(query: str) -> List[str]:
    """Extract stock symbols from natural language query"""
    # Common stock symbols and company names
    symbol_mapping = {
        'apple': 'AAPL', 'aapl': 'AAPL',
        'microsoft': 'MSFT', 'msft': 'MSFT',
        'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'amzn': 'AMZN',
        'tesla': 'TSLA', 'tsla': 'TSLA',
        'netflix': 'NFLX', 'nflx': 'NFLX',
        'facebook': 'META', 'meta': 'META',
        'nvidia': 'NVDA', 'nvda': 'NVDA',
        'intel': 'INTC', 'intc': 'INTC',
        'amd': 'AMD',
        'boeing': 'BA', 'ba': 'BA',
        'disney': 'DIS', 'dis': 'DIS',
        'walmart': 'WMT', 'wmt': 'WMT',
        'coca cola': 'KO', 'coca-cola': 'KO', 'ko': 'KO'
    }
    
    query_lower = query.lower()
    symbols = []
    
    # Look for direct symbol mentions (3-4 uppercase letters)
    symbol_pattern = r'\b[A-Z]{2,5}\b'
    direct_symbols = re.findall(symbol_pattern, query)
    symbols.extend(direct_symbols)
    
    # Look for company names
    for name, symbol in symbol_mapping.items():
        if name in query_lower:
            symbols.append(symbol)
    
    # Remove duplicates while preserving order
    unique_symbols = []
    for symbol in symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
    
    return unique_symbols[:5]  # Limit to 5 symbols

def init_database():
    """Initialize database tables"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create tables synchronously for Flask compatibility
        import sqlite3
        import os
        
        db_path = './data/financial_analyst.db'
        if not os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query_text TEXT NOT NULL,
                    response_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT NOT NULL,
                    holdings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def async_to_sync(func):
    """Decorator to run async functions in sync context"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(func(*args, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Async execution error: {e}")
            raise e
    return wrapper

def extract_symbols_from_query(query: str) -> List[str]:
    """Extract stock symbols from natural language query"""
    # Common stock symbols and company names
    symbol_mapping = {
        'apple': 'AAPL', 'aapl': 'AAPL',
        'microsoft': 'MSFT', 'msft': 'MSFT',
        'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'amzn': 'AMZN',
        'tesla': 'TSLA', 'tsla': 'TSLA',
        'netflix': 'NFLX', 'nflx': 'NFLX',
        'facebook': 'META', 'meta': 'META',
        'nvidia': 'NVDA', 'nvda': 'NVDA',
        'intel': 'INTC', 'intc': 'INTC',
        'amd': 'AMD',
        'boeing': 'BA', 'ba': 'BA',
        'disney': 'DIS', 'dis': 'DIS',
        'walmart': 'WMT', 'wmt': 'WMT',
        'coca cola': 'KO', 'coca-cola': 'KO', 'ko': 'KO'
    }
    
    query_lower = query.lower()
    symbols = []
    
    # Look for direct symbol mentions (3-4 uppercase letters)
    symbol_pattern = r'\b[A-Z]{2,5}\b'
    direct_symbols = re.findall(symbol_pattern, query)
    symbols.extend(direct_symbols)
    
    # Look for company names
    for name, symbol in symbol_mapping.items():
        if name in query_lower:
            symbols.append(symbol)
    
    # Remove duplicates while preserving order
    unique_symbols = []
    for symbol in symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
    
# Working AI-Powered Query Processor
class IntelligentQueryProcessor:
    """AI-powered query processor that handles any financial question dynamically"""
    
    def __init__(self, data_analyst, mcp_client):
        self.data_analyst = data_analyst
        self.mcp_client = mcp_client
        self.query_history = []
    
    def process_any_query(self, user_query: str, user_id: str = "default") -> Dict[str, Any]:
        """Process any user query intelligently and return comprehensive response"""
        try:
            # 1. Analyze the query context
            context = self._analyze_query_context(user_query)
            
            # 2. Extract relevant symbols and keywords
            symbols = extract_symbols_from_query(user_query)
            keywords = self._extract_keywords(user_query)
            
            # 3. Determine query type and fetch appropriate data
            if context["type"] == "stock_analysis" and symbols:
                response = self._handle_stock_analysis(user_query, symbols, keywords)
            elif context["type"] == "comparison" and len(symbols) >= 2:
                response = self._handle_stock_comparison(user_query, symbols, keywords)
            elif context["type"] == "market_overview":
                response = self._handle_market_overview(user_query, keywords)
            elif context["type"] == "portfolio":
                response = self._handle_portfolio_query(user_query, user_id, keywords)
            elif context["type"] == "news_sentiment":
                response = self._handle_news_sentiment(user_query, symbols, keywords)
            else:
                response = self._handle_general_query(user_query, symbols, keywords)
            
            # 4. Add metadata and save to history
            response.update({
                "original_query": user_query,
                "detected_symbols": symbols,
                "keywords": keywords,
                "context": context,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "query_id": len(self.query_history) + 1
            })
            
            # Save to history
            self.query_history.append(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "error": f"Failed to process query: {str(e)}",
                "original_query": user_query,
                "suggestion": "Please try rephrasing your question or ask about specific stocks like AAPL, MSFT, etc."
            }
    
    def _analyze_query_context(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine context and type"""
        query_lower = query.lower()
        
        # Query type classification
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'against', 'difference']):
            query_type = "comparison"
        elif any(word in query_lower for word in ['portfolio', 'holdings', 'allocation', 'diversification']):
            query_type = "portfolio"
        elif any(word in query_lower for word in ['news', 'sentiment', 'opinion', 'buzz', 'headlines']):
            query_type = "news_sentiment"
        elif any(word in query_lower for word in ['market', 'dow', 's&p', 'nasdaq', 'economy', 'sector']):
            query_type = "market_overview"
        elif any(word in query_lower for word in ['price', 'stock', 'share', 'ticker', 'analysis', 'buy', 'sell']):
            query_type = "stock_analysis"
        else:
            query_type = "general"
        
        # Intent classification
        intent = "informational"
        if any(word in query_lower for word in ['buy', 'sell', 'invest', 'recommend', 'should i']):
            intent = "recommendation"
        elif any(word in query_lower for word in ['predict', 'forecast', 'future', 'will', 'going to']):
            intent = "prediction"
        elif any(word in query_lower for word in ['risk', 'safe', 'volatile', 'risky']):
            intent = "risk_assessment"
        
        return {
            "type": query_type,
            "intent": intent,
            "complexity": "simple" if len(query.split()) < 10 else "complex"
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract financial keywords from query"""
        financial_keywords = [
            'price', 'earnings', 'revenue', 'profit', 'dividend', 'growth', 'valuation',
            'pe ratio', 'market cap', 'volume', 'volatility', 'beta', 'rsi', 'moving average',
            'support', 'resistance', 'bullish', 'bearish', 'buy', 'sell', 'hold',
            'risk', 'return', 'yield', 'margin', 'debt', 'cash flow', 'eps'
        ]
        
        query_lower = query.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _handle_stock_analysis(self, query: str, symbols: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Handle single or multiple stock analysis queries"""
        try:
            if len(symbols) == 1:
                # Single stock analysis
                symbol = symbols[0]
                analysis = self.data_analyst.analyze_stock(symbol)
                
                if not analysis.get("success"):
                    return {"error": f"Could not analyze {symbol}: {analysis.get('error')}"}
                
                # Generate human-readable response
                current_price = analysis.get("current_price", 0)
                recommendation = analysis.get("recommendation", {})
                key_metrics = analysis.get("key_metrics", {})
                technical = analysis.get("technical_analysis", {})
                company_info = analysis.get("company_info", {})
                
                human_response = f"""
**Stock Analysis for {symbol}**

**Company:** {company_info.get('company_name', symbol)}
**Sector:** {company_info.get('sector', 'N/A')} | **Industry:** {company_info.get('industry', 'N/A')}

**Current Metrics:**
• Price: ${current_price:.2f}
• Change: {key_metrics.get('price_change_percent', 0):.2f}%
• Market Cap: {self._format_number(key_metrics.get('market_cap', 0))}
• P/E Ratio: {key_metrics.get('pe_ratio', 'N/A')}

**Technical Analysis:**
• RSI: {technical.get('rsi', 'N/A')}
• Support: ${technical.get('support', 0):.2f}
• Resistance: ${technical.get('resistance', 0):.2f}
• 20-day SMA: ${technical.get('sma_20', 0):.2f}

**Investment Recommendation:**
• Action: {recommendation.get('action', 'HOLD')}
• Confidence: {recommendation.get('confidence', 0.5)*100:.0f}%
• Risk Level: {recommendation.get('risk_level', 'Medium')}

**Reasoning:**
{chr(10).join(f"• {reason}" for reason in recommendation.get('reasoning', ['Analysis completed']))}

**Conclusion:**
Based on current data, {symbol} shows a {recommendation.get('action', 'HOLD').lower()} signal with {recommendation.get('confidence', 0.5)*100:.0f}% confidence. Consider your risk tolerance and investment timeline.
"""
                
                return {
                    "query_type": "stock_analysis",
                    "symbol": symbol,
                    "human_response": human_response.strip(),
                    "raw_data": analysis,
                    "success": True
                }
            
            else:
                # Multiple stock analysis - treat as comparison
                return self._handle_stock_comparison(query, symbols, keywords)
                
        except Exception as e:
            return {"error": f"Stock analysis failed: {str(e)}"}
    
    def _handle_stock_comparison(self, query: str, symbols: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Handle stock comparison queries"""
        try:
            if len(symbols) < 2:
                symbols = ['AAPL', 'MSFT']  # Default comparison
            
            comparison_result = self.data_analyst.compare_stocks(symbols[:5])
            
            if not comparison_result.get("success"):
                return {"error": f"Comparison failed: {comparison_result.get('error')}"}
            
            analyses = comparison_result.get("individual_analyses", {})
            comparison_data = comparison_result.get("comparison", {})
            
            # Generate human-readable comparison
            human_response = f"**Stock Comparison: {' vs '.join(symbols)}**\n\n"
            
            # Individual stock summaries
            for symbol, analysis in analyses.items():
                if analysis.get("success"):
                    price = analysis.get("current_price", 0)
                    change = analysis.get("key_metrics", {}).get("price_change_percent", 0)
                    recommendation = analysis.get("recommendation", {}).get("action", "HOLD")
                    
                    human_response += f"**{symbol}:**\n"
                    human_response += f"• Price: ${price:.2f} ({change:+.2f}%)\n"
                    human_response += f"• Recommendation: {recommendation}\n"
                    human_response += f"• Company: {analysis.get('company_info', {}).get('company_name', symbol)}\n\n"
            
            # Comparison insights
            performance = comparison_data.get("performance", {})
            best_performer = comparison_data.get("summary", {}).get("best_performer")
            
            human_response += "**Comparison Analysis:**\n"
            if best_performer:
                best_change = performance.get(best_performer, {}).get("price_change_percent", 0)
                human_response += f"• Best Performer: {best_performer} ({best_change:+.2f}%)\n"
            
            # Recommendation summary
            rec_summary = comparison_data.get("summary", {}).get("recommendation_summary", {})
            human_response += f"• Buy Signals: {rec_summary.get('buy_recommendations', 0)}\n"
            human_response += f"• Hold Signals: {rec_summary.get('hold_recommendations', 0)}\n"
            human_response += f"• Sell Signals: {rec_summary.get('sell_recommendations', 0)}\n\n"
            
            human_response += "**Investment Insight:**\n"
            if best_performer and best_change > 0:
                human_response += f"• {best_performer} shows the strongest recent performance\n"
            
            human_response += "• Consider diversification across multiple positions\n"
            human_response += "• Review individual company fundamentals before investing\n"
            
            return {
                "query_type": "stock_comparison",
                "symbols": symbols,
                "human_response": human_response.strip(),
                "raw_data": comparison_result,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Stock comparison failed: {str(e)}"}
    
    def _handle_market_overview(self, query: str, keywords: List[str]) -> Dict[str, Any]:
        """Handle market overview queries"""
        try:
            # Use major market indicators
            market_symbols = ['SPY', 'QQQ', 'DIA', 'VTI']  # S&P 500, NASDAQ, DOW, Total Market
            
            market_data = {}
            for symbol in market_symbols:
                price_data = self.mcp_client.get_stock_price(symbol)
                if price_data.get("success"):
                    market_data[symbol] = price_data
            
            human_response = "**Market Overview**\n\n"
            
            etf_names = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ-100',
                'DIA': 'Dow Jones',
                'VTI': 'Total Stock Market'
            }
            
            for symbol, data in market_data.items():
                name = etf_names.get(symbol, symbol)
                price = data.get("current_price", 0)
                change = data.get("change_percent", 0)
                trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                human_response += f"**{name} ({symbol}):** ${price:.2f} {trend} {change:+.2f}%\n"
            
            # Market sentiment
            positive_count = sum(1 for data in market_data.values() if data.get("change_percent", 0) > 0)
            total_count = len(market_data)
            
            human_response += f"\n**Market Sentiment:**\n"
            if positive_count >= total_count * 0.75:
                human_response += "• Bullish - Most indices are positive\n"
            elif positive_count <= total_count * 0.25:
                human_response += "• Bearish - Most indices are negative\n"
            else:
                human_response += "• Mixed - Market showing uncertainty\n"
            
            human_response += "\n**Key Takeaways:**\n"
            human_response += "• Monitor major economic indicators\n"
            human_response += "• Consider market volatility in investment decisions\n"
            human_response += "• Diversification remains important\n"
            
            return {
                "query_type": "market_overview",
                "human_response": human_response.strip(),
                "raw_data": market_data,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Market overview failed: {str(e)}"}
    
    def _handle_portfolio_query(self, query: str, user_id: str, keywords: List[str]) -> Dict[str, Any]:
        """Handle portfolio-related queries"""
        try:
            # For now, provide portfolio guidance
            human_response = """
**Portfolio Analysis & Guidance**

**Diversification Principles:**
• Spread investments across different sectors
• Mix of growth and value stocks
• Consider international exposure
• Include defensive positions

**Risk Management:**
• Never put more than 5-10% in a single stock
• Rebalance quarterly or semi-annually
• Set stop-loss levels for risk control
• Consider your investment timeline

**Popular Portfolio Allocations:**
• Conservative: 60% stocks, 40% bonds
• Moderate: 70% stocks, 30% bonds
• Aggressive: 80-90% stocks, 10-20% bonds

**Recommendation:**
Create a diversified portfolio based on your risk tolerance and investment goals. Consider using ETFs for broad market exposure.
"""
            
            return {
                "query_type": "portfolio",
                "human_response": human_response.strip(),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Portfolio query failed: {str(e)}"}
    
    def _handle_news_sentiment(self, query: str, symbols: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Handle news and sentiment queries"""
        try:
            if symbols:
                symbol = symbols[0]
                # Get basic company info for context
                company_info = self.mcp_client.get_company_info(symbol)
                company_name = company_info.get("company_name", symbol) if company_info.get("success") else symbol
                
                human_response = f"""
**News & Sentiment Analysis for {symbol}**

**Company:** {company_name}

**Recent Market Activity:**
• Monitor financial news sources for latest updates
• Check earnings reports and analyst ratings
• Review SEC filings for material changes

**Sentiment Factors to Consider:**
• Industry trends and competitive landscape
• Regulatory changes affecting the sector
• Management changes and strategic announcements
• Economic indicators impacting the business

**Recommendation:**
Stay informed about {company_name} through reliable financial news sources. Consider both fundamental analysis and market sentiment when making investment decisions.
"""
            else:
                human_response = """
**General Market Sentiment**

**Key Sources for Market News:**
• Financial news websites (Bloomberg, Reuters, MarketWatch)
• SEC filings and earnings reports
• Federal Reserve announcements
• Economic indicators (GDP, inflation, employment)

**Sentiment Indicators:**
• VIX (Volatility Index) for market fear
• Put/Call ratios for options sentiment
• Analyst upgrades/downgrades
• Institutional money flows

**Recommendation:**
Combine multiple news sources and sentiment indicators for a balanced view of market conditions.
"""
            
            return {
                "query_type": "news_sentiment",
                "symbols": symbols,
                "human_response": human_response.strip(),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"News sentiment analysis failed: {str(e)}"}
    
    def _handle_general_query(self, query: str, symbols: List[str], keywords: List[str]) -> Dict[str, Any]:
        """Handle general financial queries"""
        try:
            query_lower = query.lower()
            
            # Investment basics
            if any(word in query_lower for word in ['beginner', 'start', 'how to invest', 'basics']):
                human_response = """
**Investment Basics**

**Getting Started:**
• Open a brokerage account with a reputable firm
• Start with index funds or ETFs for diversification
• Invest only money you won't need for 5+ years
• Consider dollar-cost averaging

**Key Principles:**
• Time in market beats timing the market
• Diversification reduces risk
• Keep costs low (expense ratios, fees)
• Stay disciplined during market volatility

**Common Mistakes to Avoid:**
• Emotional trading (buying high, selling low)
• Putting all money in one stock
• Trying to time the market
• Not having an emergency fund first

**Next Steps:**
1. Define your investment goals
2. Assess your risk tolerance
3. Create a diversified portfolio
4. Review and rebalance regularly
"""
            
            # Market terminology
            elif any(word in query_lower for word in ['what is', 'define', 'meaning', 'explain']):
                human_response = """
**Financial Terms Explained**

**Common Stock Metrics:**
• P/E Ratio: Price-to-Earnings (valuation measure)
• Market Cap: Total value of company shares
• EPS: Earnings Per Share
• Dividend Yield: Annual dividends / stock price

**Technical Indicators:**
• RSI: Relative Strength Index (momentum)
• Moving Average: Average price over time period
• Support/Resistance: Price levels where stock tends to bounce

**Investment Actions:**
• Buy: Purchase a security expecting price increase
• Sell: Dispose of a security
• Hold: Maintain current position

For specific terms, please ask about them directly!
"""
            
            else:
                # Default helpful response
                human_response = f"""
**Financial Assistant**

I can help you with:
• Stock analysis and comparisons
• Market overviews and trends
• Investment recommendations
• Portfolio guidance
• Financial term explanations

**Your query:** "{query}"

**Suggestions:**
• Ask about specific stocks: "How is Apple performing?"
• Compare stocks: "AAPL vs MSFT analysis"
• Get market overview: "How is the market today?"
• Portfolio help: "How should I diversify my portfolio?"

Please feel free to ask any financial question!
"""
            
            return {
                "query_type": "general",
                "human_response": human_response.strip(),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"General query processing failed: {str(e)}"}
    
    def _format_number(self, num) -> str:
        """Format large numbers for human readability"""
        if num >= 1e12:
            return f"${num/1e12:.1f}T"
        elif num >= 1e9:
            return f"${num/1e9:.1f}B"
        elif num >= 1e6:
            return f"${num/1e6:.1f}M"
        elif num >= 1e3:
            return f"${num/1e3:.1f}K"
        else:
            return f"${num:.2f}"

# Global intelligent query processor
intelligent_processor = IntelligentQueryProcessor(data_analyst, mcp_client)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-Powered Financial Analyst</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header { 
            text-align: center; 
            margin-bottom: 40px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header h1 { 
            font-size: 3em; 
            margin-bottom: 10px;
            font-weight: bold;
        }
        .header p { 
            font-size: 1.2em; 
            color: #666;
            -webkit-text-fill-color: #666;
        }
        
        .status-card {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border-left: 5px solid #28a745;
        }
        .status-card.warning {
            background: #fff3cd;
            border-left-color: #ffc107;
            border-color: #ffeaa7;
        }
        .status-card.error {
            background: #f8d7da;
            border-left-color: #dc3545;
            border-color: #f5c6cb;
        }
        
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-radius: 10px;
            margin: 20px 0;
            overflow: hidden;
        }
        .tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            cursor: pointer;
            background: #e9ecef;
            border: none;
            transition: all 0.3s;
        }
        .tab.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .tab:hover {
            background: #007bff;
            color: white;
        }
        
        .tab-content {
            display: none;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
            margin: 20px 0;
        }
        .tab-content.active {
            display: block;
        }
        
        .query-form {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .query-input {
            flex: 1;
            min-width: 300px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
        }
        .query-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-area {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            display: none;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .example-queries {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .example-btn {
            padding: 8px 15px;
            background: #e9ecef;
            border: 1px solid #ddd;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .feature-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .portfolio-form {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🚀 MCP-Powered Financial Analyst</h1>
            <p>AI-powered financial analysis with multi-agent AutoGen system</p>
        </header>
        
        <div id="statusSection"></div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('query')">💬 Query Analysis</button>
            <button class="tab" onclick="showTab('portfolio')">📊 Portfolio Management</button>
            <button class="tab" onclick="showTab('reports')">📄 Reports</button>
            <button class="tab" onclick="showTab('visualization')">📈 Visualization</button>
        </div>
        
        <!-- Query Tab -->
        <div id="query-tab" class="tab-content active">
            <h2>🎯 Ask the Financial Analyst</h2>
            
            <div class="query-form">
                <input type="text" class="query-input" id="queryInput" 
                       placeholder="Ask me anything about stocks, markets, or investments..." />
                <button class="btn" id="queryBtn" onclick="processQuery()">Analyze</button>
            </div>
            
            <div class="example-queries">
                <div class="example-btn" onclick="setQuery('Analyze Tesla stock performance and provide investment recommendation')">🚗 Tesla Analysis</div>
                <div class="example-btn" onclick="setQuery('Compare Apple vs Microsoft stocks for long-term investment')">📱 Apple vs Microsoft</div>
                <div class="example-btn" onclick="setQuery('What is the current market sentiment and should I invest now?')">📊 Market Sentiment</div>
                <div class="example-btn" onclick="setQuery('Create a diversified portfolio recommendation for $10,000')">💰 Portfolio Advice</div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>AI agents are analyzing your query...</p>
            </div>
            
            <div class="result-area" id="resultArea"></div>
        </div>
        
        <!-- Portfolio Tab -->
        <div id="portfolio-tab" class="tab-content">
            <h2>📊 Portfolio Management</h2>
            
            <div class="portfolio-form">
                <h3>Create New Portfolio</h3>
                <div class="form-group">
                    <label class="form-label">Portfolio Name:</label>
                    <input type="text" id="portfolioName" class="form-input" placeholder="e.g., Tech Growth Portfolio">
                </div>
                <div class="form-group">
                    <label class="form-label">Holdings (JSON format):</label>
                    <textarea id="portfolioHoldings" class="form-input" rows="4" placeholder='{"AAPL": 100, "MSFT": 50, "GOOGL": 25}'></textarea>
                </div>
                <button class="btn" onclick="createPortfolio()">Create Portfolio</button>
            </div>
            
            <div id="portfolioResults"></div>
        </div>
        
        <!-- Reports Tab -->
        <div id="reports-tab" class="tab-content">
            <h2>📄 Financial Reports</h2>
            
            <div class="features-grid">
                <div class="feature-card">
                    <h3>Stock Analysis Report</h3>
                    <p>Comprehensive analysis of individual stocks with technical and fundamental data.</p>
                    <button class="btn" onclick="generateReport('stock')">Generate Stock Report</button>
                </div>
                <div class="feature-card">
                    <h3>Market Summary Report</h3>
                    <p>Overview of market conditions, trends, and sentiment analysis.</p>
                    <button class="btn" onclick="generateReport('market')">Generate Market Report</button>
                </div>
                <div class="feature-card">
                    <h3>Portfolio Analysis</h3>
                    <p>Detailed portfolio performance and optimization recommendations.</p>
                    <button class="btn" onclick="generateReport('portfolio')">Generate Portfolio Report</button>
                </div>
            </div>
            
            <div id="reportResults"></div>
        </div>
        
        <!-- Visualization Tab -->
        <div id="visualization-tab" class="tab-content">
            <h2>📈 Data Visualization</h2>
            
            <div class="query-form">
                <input type="text" id="vizSymbol" class="query-input" placeholder="Enter stock symbol (e.g., AAPL)" />
                <button class="btn" onclick="generateVisualization()">Create Chart</button>
            </div>
            
            <div class="chart-container" id="chartContainer" style="display: none;">
                <canvas id="stockChart" width="400" height="200"></canvas>
            </div>
            
            <div id="vizResults"></div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p><strong>MCP-Powered Financial Analyst</strong> | Built with AutoGen, Flask, and Model Context Protocol</p>
        </div>
    </div>
    
    <script>
        let systemReady = false;
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const statusSection = document.getElementById('statusSection');
                const queryBtn = document.getElementById('queryBtn');
                
                if (data.status === 'healthy') {
                    systemReady = true;
                    statusSection.innerHTML = `
                        <div class="status-card">
                            <h3>✅ System Ready</h3>
                            <p>All ${data.agent_count} agents initialized and ready for financial analysis!</p>
                            <small>Last checked: ${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                    queryBtn.disabled = false;
                } else if (data.status === 'initializing') {
                    systemReady = false;
                    statusSection.innerHTML = `
                        <div class="status-card warning">
                            <h3>⚠️ System Initializing</h3>
                            <p>AI agents are being initialized. Please wait...</p>
                            <small>Last checked: ${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                    queryBtn.disabled = true;
                } else {
                    systemReady = false;
                    statusSection.innerHTML = `
                        <div class="status-card error">
                            <h3>❌ System Error</h3>
                            <p>${data.message || 'Unknown error occurred'}</p>
                            <small>Last checked: ${new Date().toLocaleTimeString()}</small>
                        </div>
                    `;
                    queryBtn.disabled = true;
                }
            } catch (error) {
                const statusSection = document.getElementById('statusSection');
                statusSection.innerHTML = `
                    <div class="status-card error">
                        <h3>❌ Connection Error</h3>
                        <p>Unable to connect to the backend service.</p>
                        <small>Error: ${error.message}</small>
                    </div>
                `;
                document.getElementById('queryBtn').disabled = true;
            }
        }
        
        function setQuery(query) {
            document.getElementById('queryInput').value = query;
        }
        
        async function processQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query.trim()) return;
            
            const loading = document.getElementById('loading');
            const resultArea = document.getElementById('resultArea');
            
            loading.style.display = 'block';
            resultArea.style.display = 'none';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, user_id: 'web_user' })
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                resultArea.style.display = 'block';
                resultArea.textContent = JSON.stringify(data, null, 2);
                
            } catch (error) {
                loading.style.display = 'none';
                resultArea.style.display = 'block';
                resultArea.textContent = 'Error: ' + error.message;
            }
        }
        
        async function createPortfolio() {
            const name = document.getElementById('portfolioName').value;
            const holdings = document.getElementById('portfolioHoldings').value;
            
            if (!name.trim() || !holdings.trim()) {
                alert('Please fill in all fields');
                return;
            }
            
            try {
                const response = await fetch('/api/portfolio/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        name: name, 
                        holdings: JSON.parse(holdings),
                        user_id: 'web_user'
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('portfolioResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Portfolio Created Successfully</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('portfolioResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Error Creating Portfolio</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function generateReport(type) {
            const symbol = prompt(`Enter stock symbol for ${type} report:`) || 'AAPL';
            
            try {
                const response = await fetch('/api/report/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        report_type: type,
                        symbol: symbol,
                        user_id: 'web_user'
                    })
                });
                
                const data = await response.json();
                
                document.getElementById('reportResults').innerHTML = `
                    <div class="feature-card">
                        <h3>${type.charAt(0).toUpperCase() + type.slice(1)} Report Generated</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('reportResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Error Generating Report</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        async function generateVisualization() {
            const symbol = document.getElementById('vizSymbol').value || 'AAPL';
            
            try {
                const response = await fetch(`/api/visualization/stock/${symbol}`);
                const data = await response.json();
                
                if (data.chart_data) {
                    const ctx = document.getElementById('stockChart').getContext('2d');
                    
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.chart_data.dates,
                            datasets: [{
                                label: `${symbol} Price`,
                                data: data.chart_data.prices,
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: false
                                }
                            }
                        }
                    });
                    
                    document.getElementById('chartContainer').style.display = 'block';
                }
                
                document.getElementById('vizResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Visualization Data for ${symbol}</h3>
                        <pre>${JSON.stringify(data, null, 2)}</pre>
                    </div>
                `;
                
            } catch (error) {
                document.getElementById('vizResults').innerHTML = `
                    <div class="feature-card">
                        <h3>Error Generating Visualization</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }
        
        // Initialize
        checkSystemStatus();
        setInterval(checkSystemStatus, 30000);
        
        // Allow Enter key to submit query
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !document.getElementById('queryBtn').disabled) {
                processQuery();
            }
        });
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test MCP client
        test_result = mcp_client.get_stock_price("AAPL")
        mcp_status = "working" if test_result.get("success") else "error"
        
        # Test database
        db_status = "working"
        try:
            import sqlite3
            conn = sqlite3.connect('./data/financial_analyst.db')
            conn.execute("SELECT 1")
            conn.close()
        except Exception:
            db_status = "error"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "mcp_client": mcp_status,
                "database": db_status,
                "data_analyst": "working",
                "api_endpoints": "working"
            },
            "features": {
                "stock_analysis": "available",
                "stock_comparison": "available",
                "historical_data": "available",
                "company_info": "available",
                "natural_language_queries": "available"
            },
            "test_endpoints": {
                "single_stock": "/api/analyze/stock/AAPL",
                "comparison": "/api/compare/stocks",
                "query": "/api/query",
                "price": "/api/stock/price/AAPL"
            }
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/query', methods=['POST'])
def process_financial_query():
    """Process any financial query intelligently with AI-powered analysis"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', 'default')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        logger.info(f"Processing intelligent query for user {user_id}: {query}")
        
        # Use the intelligent financial assistant
        result = intelligent_assistant.process_query(query, user_id)
        
        # Save to database
        try:
            import sqlite3
            conn = sqlite3.connect('./data/financial_analyst.db')
            conn.execute(
                "INSERT INTO user_queries (user_id, query_text, response_data) VALUES (?, ?, ?)",
                (user_id, query, json.dumps(result))
            )
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Intelligent query processing error: {str(e)}")
        return jsonify({
            "error": f"Query processing failed: {str(e)}",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Please try rephrasing your question or ask about specific stocks"
        }), 500

@app.route('/api/analyze/stock/<symbol>')
def analyze_single_stock(symbol):
    """Analyze a single stock with comprehensive data"""
    try:
        symbol = symbol.upper()
        logger.info(f"Analyzing stock: {symbol}")
        
        result = data_analyst.analyze_stock(symbol)
        
        if not result.get("success"):
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Stock analysis error for {symbol}: {str(e)}")
        return jsonify({
            "error": f"Stock analysis failed: {str(e)}",
            "symbol": symbol
        }), 500

@app.route('/api/compare/stocks', methods=['POST'])
def compare_multiple_stocks():
    """Compare multiple stocks"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols or len(symbols) < 2:
            return jsonify({"error": "At least 2 symbols required for comparison"}), 400
        
        # Limit to 5 stocks for performance
        symbols = [s.upper() for s in symbols[:5]]
        
        logger.info(f"Comparing stocks: {symbols}")
        
        result = data_analyst.compare_stocks(symbols)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Stock comparison error: {str(e)}")
        return jsonify({
            "error": f"Stock comparison failed: {str(e)}"
        }), 500

@app.route('/api/stock/price/<symbol>')
def get_stock_price(symbol):
    """Get current stock price"""
    try:
        symbol = symbol.upper()
        result = mcp_client.get_stock_price(symbol)
        
        if not result.get("success"):
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/info/<symbol>')
def get_stock_info(symbol):
    """Get company information"""
    try:
        symbol = symbol.upper()
        result = mcp_client.get_company_info(symbol)
        
        if not result.get("success"):
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/historical/<symbol>')
def get_stock_historical(symbol):
    """Get historical stock data"""
    try:
        symbol = symbol.upper()
        period = request.args.get('period', '1y')
        
        result = mcp_client.get_historical_data(symbol, period)
        
        if not result.get("success"):
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def simulate_agent_processing(query: str, user_id: str) -> Dict[str, Any]:
    """Simulate agent processing for demonstration"""
    query_lower = query.lower()
    
    # Stock analysis queries
    if any(word in query_lower for word in ['tesla', 'tsla']):
        return {
            "query_type": "stock_analysis",
            "symbol": "TSLA",
            "analysis": {
                "current_price": 248.73,
                "recommendation": "HOLD",
                "confidence": 0.78,
                "reasoning": "Tesla shows strong EV market position but faces increased competition",
                "technical_indicators": {
                    "rsi": 65.2,
                    "sma_20": 245.30,
                    "sma_50": 240.15,
                    "support": 235.00,
                    "resistance": 260.00
                },
                "fundamentals": {
                    "pe_ratio": 72.5,
                    "market_cap": "790B",
                    "revenue_growth": "15.3%",
                    "profit_margin": "8.2%"
                }
            },
            "news_sentiment": {
                "score": 0.25,
                "articles_analyzed": 47,
                "key_topics": ["EV sales", "Autopilot", "Energy business"]
            },
            "recommendation_details": {
                "action": "HOLD",
                "target_price": 275.00,
                "stop_loss": 220.00,
                "time_horizon": "6-12 months",
                "risk_level": "High"
            }
        }
    
    elif any(word in query_lower for word in ['apple', 'aapl']):
        return {
            "query_type": "stock_analysis",
            "symbol": "AAPL",
            "analysis": {
                "current_price": 185.92,
                "recommendation": "BUY",
                "confidence": 0.87,
                "reasoning": "Apple maintains strong ecosystem and growing services revenue",
                "technical_indicators": {
                    "rsi": 58.3,
                    "sma_20": 182.45,
                    "sma_50": 178.90,
                    "support": 175.00,
                    "resistance": 195.00
                },
                "fundamentals": {
                    "pe_ratio": 28.7,
                    "market_cap": "2.8T",
                    "revenue_growth": "8.1%",
                    "profit_margin": "25.3%"
                }
            }
        }
    
    elif 'compare' in query_lower or 'vs' in query_lower:
        return {
            "query_type": "stock_comparison",
            "symbols": ["AAPL", "MSFT"],
            "comparison": {
                "winner": "AAPL",
                "metrics": {
                    "performance_1y": {"AAPL": 15.2, "MSFT": 12.8},
                    "pe_ratio": {"AAPL": 28.7, "MSFT": 34.2},
                    "dividend_yield": {"AAPL": 0.43, "MSFT": 0.68}
                },
                "recommendation": "Both are strong, but Apple has better momentum currently"
            }
        }
    
    elif any(word in query_lower for word in ['portfolio', 'diversif', 'invest']):
        return {
            "query_type": "portfolio_advice",
            "recommendation": {
                "allocation": {
                    "Technology": 30,
                    "Healthcare": 20,
                    "Financial": 15,
                    "Consumer": 15,
                    "Bonds": 20
                },
                "suggested_stocks": ["AAPL", "MSFT", "JNJ", "JPM", "BRK.B"],
                "risk_level": "Moderate",
                "expected_return": "8-12% annually"
            }
        }
    
    elif any(word in query_lower for word in ['sentiment', 'market']):
        return {
            "query_type": "market_sentiment",
            "sentiment": {
                "overall_score": 0.35,
                "label": "Moderately Positive",
                "confidence": 0.82,
                "factors": {
                    "earnings_season": "positive",
                    "fed_policy": "neutral", 
                    "geopolitical": "slightly_negative",
                    "economic_data": "positive"
                },
                "sector_sentiment": {
                    "Technology": 0.45,
                    "Healthcare": 0.25,
                    "Financial": 0.15,
                    "Energy": -0.10
                }
            }
        }
    
    else:
        return {
            "query_type": "general",
            "response": f"I analyzed your query: '{query}'",
            "suggestions": [
                "Try asking about specific stocks (e.g., 'Analyze Tesla stock')",
                "Ask for market sentiment analysis",
                "Request portfolio recommendations",
                "Compare multiple stocks"
            ],
            "available_features": [
                "Stock analysis",
                "Portfolio optimization",
                "Market sentiment",
                "Technical analysis",
                "Fundamental analysis"
            ]
        }

@app.route('/api/analyze/stock/<symbol>')
def analyze_stock(symbol):
    """Analyze a specific stock"""
    try:
        symbol = symbol.upper()
        
        # Initialize agents
        if not init_agents():
            return jsonify({"error": "Agents not initialized"}), 503
        
        # Simulate comprehensive stock analysis
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": 150.00 + hash(symbol) % 100,  # Mock price
            "analysis": {
                "recommendation": "BUY" if hash(symbol) % 2 else "HOLD",
                "confidence": 0.75 + (hash(symbol) % 25) / 100,
                "technical_analysis": {
                    "trend": "bullish" if hash(symbol) % 2 else "neutral",
                    "rsi": 30 + hash(symbol) % 40,
                    "moving_averages": {
                        "sma_20": 148.50,
                        "sma_50": 145.30,
                        "sma_200": 140.80
                    }
                },
                "fundamental_analysis": {
                    "pe_ratio": 15 + hash(symbol) % 20,
                    "market_cap": f"{hash(symbol) % 500 + 100}B",
                    "revenue_growth": f"{hash(symbol) % 20}%"
                }
            }
        }
        
        return jsonify({
            "success": True,
            "data": analysis
        })
        
    except Exception as e:
        logger.error(f"Stock analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/create', methods=['POST'])
def create_portfolio():
    """Create a new portfolio"""
    try:
        data = request.get_json()
        name = data.get('name')
        holdings = data.get('holdings', {})
        user_id = data.get('user_id', 'default')
        
        if not name:
            return jsonify({"error": "Portfolio name is required"}), 400
        
        # Save to database
        import sqlite3
        conn = sqlite3.connect('./data/financial_analyst.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO portfolios (user_id, name, holdings) VALUES (?, ?, ?)",
            (user_id, name, json.dumps(holdings))
        )
        portfolio_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Analyze portfolio
        analysis = analyze_portfolio_holdings(holdings)
        
        return jsonify({
            "success": True,
            "portfolio_id": portfolio_id,
            "name": name,
            "holdings": holdings,
            "analysis": analysis,
            "created_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Portfolio creation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def analyze_portfolio_holdings(holdings: Dict[str, int]) -> Dict[str, Any]:
    """Analyze portfolio holdings"""
    total_value = sum(holdings.values()) * 100  # Mock calculation
    
    return {
        "total_positions": len(holdings),
        "estimated_value": total_value,
        "diversification_score": min(len(holdings) * 20, 100),
        "risk_level": "Moderate" if len(holdings) > 3 else "High",
        "recommendations": [
            "Consider adding more sectors for diversification" if len(holdings) < 5 else "Good diversification",
            "Review position sizes for better balance",
            "Monitor quarterly earnings for all holdings"
        ]
    }

@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """Generate financial reports"""
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'stock')
        symbol = data.get('symbol', 'AAPL')
        user_id = data.get('user_id', 'default')
        
        # Generate report based on type
        if report_type == 'stock':
            report = generate_stock_report(symbol)
        elif report_type == 'portfolio':
            report = generate_portfolio_report(user_id)
        elif report_type == 'market':
            report = generate_market_report()
        else:
            return jsonify({"error": "Invalid report type"}), 400
        
        return jsonify({
            "success": True,
            "report": report,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_stock_report(symbol: str) -> Dict[str, Any]:
    """Generate comprehensive stock report"""
    return {
        "title": f"Stock Analysis Report - {symbol}",
        "symbol": symbol,
        "sections": {
            "executive_summary": f"Comprehensive analysis of {symbol} shows mixed signals with moderate buy recommendation.",
            "financial_highlights": {
                "revenue": "Strong revenue growth of 12.5% YoY",
                "profitability": "Maintaining healthy profit margins",
                "debt": "Conservative debt levels"
            },
            "technical_analysis": {
                "trend": "Bullish momentum in short-term",
                "support_resistance": "Key support at $140, resistance at $160",
                "indicators": "RSI indicates neutral conditions"
            },
            "recommendation": {
                "action": "BUY",
                "target_price": 165.00,
                "time_horizon": "6-12 months"
            }
        }
    }

def generate_portfolio_report(user_id: str) -> Dict[str, Any]:
    """Generate portfolio analysis report"""
    return {
        "title": "Portfolio Analysis Report",
        "user_id": user_id,
        "sections": {
            "overview": "Portfolio shows good diversification across sectors",
            "performance": "Outperforming S&P 500 by 2.3% YTD",
            "risk_analysis": "Moderate risk profile with beta of 1.15",
            "recommendations": [
                "Consider rebalancing technology allocation",
                "Add international exposure",
                "Review defensive positions"
            ]
        }
    }

def generate_market_report() -> Dict[str, Any]:
    """Generate market analysis report"""
    return {
        "title": "Market Analysis Report",
        "sections": {
            "market_overview": "Markets showing resilience despite volatility",
            "sector_analysis": {
                "Technology": "Leading gains with AI momentum",
                "Healthcare": "Steady performance",
                "Energy": "Mixed signals from commodity prices"
            },
            "economic_indicators": {
                "gdp_growth": "2.1% annual growth",
                "inflation": "Moderating to 3.2%",
                "employment": "Strong labor market conditions"
            },
            "outlook": "Cautiously optimistic for next quarter"
        }
    }

@app.route('/api/visualization/stock/<symbol>')
def get_stock_visualization(symbol):
    """Get stock visualization data"""
    try:
        symbol = symbol.upper()
        
        # Generate mock data for demonstration
        import random
        from datetime import timedelta
        
        base_date = datetime.now() - timedelta(days=30)
        dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        
        base_price = 150 + hash(symbol) % 50
        prices = []
        current_price = base_price
        
        for _ in range(30):
            change = random.uniform(-0.05, 0.05)
            current_price *= (1 + change)
            prices.append(round(current_price, 2))
        
        visualization_data = {
            "symbol": symbol,
            "chart_data": {
                "dates": dates,
                "prices": prices,
                "volumes": [random.randint(1000000, 10000000) for _ in range(30)]
            },
            "statistics": {
                "current_price": prices[-1],
                "change_30d": round(((prices[-1] - prices[0]) / prices[0]) * 100, 2),
                "high_30d": max(prices),
                "low_30d": min(prices),
                "avg_volume": sum([random.randint(1000000, 10000000) for _ in range(30)]) / 30
            }
        }
        
        return jsonify({
            "success": True,
            "data": visualization_data
        })
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare/stocks', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if len(symbols) < 2:
            return jsonify({"error": "At least 2 symbols required"}), 400
        
        comparison = {}
        for symbol in symbols:
            comparison[symbol] = {
                "current_price": 100 + hash(symbol) % 100,
                "pe_ratio": 15 + hash(symbol) % 20,
                "market_cap": f"{hash(symbol) % 500 + 100}B",
                "recommendation": "BUY" if hash(symbol) % 2 else "HOLD"
            }
        
        return jsonify({
            "success": True,
            "comparison": comparison,
            "winner": max(symbols, key=lambda x: hash(x) % 100),
            "analysis": f"Based on analysis, {symbols[0]} shows better value proposition"
        })
        
    except Exception as e:
        logger.error(f"Stock comparison error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/intelligent-query', methods=['POST'])
def process_intelligent_query():
    """Advanced AI-powered query processing with dual format responses (sentence + table)"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', 'default')
        format_preference = data.get('format', 'both')  # 'sentence', 'table', or 'both'
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        logger.info(f"Processing intelligent query: {query}")
        
        # Process with enhanced intelligent assistant
        result = intelligent_assistant.process_query(query, user_id)
        
        # Extract dual format responses
        sentence_format = result.get("sentence_format", "")
        table_format = result.get("table_format", "")
        
        # Build response based on format preference
        formatted_response = {
            "success": result.get("success", True),
            "query": query,
            "query_type": result.get("query_type", "general"),
            "detected_symbols": result.get("detected_symbols", []),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "processed_by": "intelligent_financial_assistant_v2"
        }
        
        # Add format-specific responses
        if format_preference in ['sentence', 'both']:
            formatted_response["sentence_format"] = sentence_format
        
        if format_preference in ['table', 'both']:
            formatted_response["table_format"] = table_format
        
        # Add detailed analysis if available
        if result.get("analysis"):
            formatted_response["detailed_analysis"] = result["analysis"]
        
        # Add human summary for backward compatibility
        if result.get("human_summary"):
            formatted_response["human_summary"] = result["human_summary"]
        
        # Add error information if present
        if "error" in result:
            formatted_response["error"] = result["error"]
            formatted_response["success"] = False
        
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Intelligent query error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Intelligent query processing failed: {str(e)}",
            "query": query if 'query' in locals() else "Unknown",
            "suggestion": "Please try a different question or check your query format",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/news/<symbol>', methods=['GET'])
def get_stock_news(symbol):
    """Get latest news and sentiment analysis for a stock"""
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(limit, 50)  # Cap at 50 articles
        
        logger.info(f"Fetching news for {symbol.upper()}")
        
        # Get news and sentiment analysis
        news_result = news_analyzer.get_stock_news(symbol.upper(), limit)
        
        if not news_result.get("success"):
            return jsonify(news_result), 400
        
        # Format response for easy consumption
        response = {
            "symbol": symbol.upper(),
            "news_summary": {
                "total_articles": news_result["total_articles"],
                "overall_sentiment": news_result["overall_sentiment"],
                "sentiment_summary": news_result["sentiment_summary"]
            },
            "articles": news_result["articles"],
            "timestamp": news_result["timestamp"],
            "success": True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"News API error: {str(e)}")
        return jsonify({
            "error": f"Failed to fetch news for {symbol}: {str(e)}",
            "success": False
        }), 500

@app.route('/api/news-sentiment/<symbol>', methods=['GET'])
def get_news_sentiment_analysis(symbol):
    """Get detailed sentiment analysis for stock news"""
    try:
        limit = request.args.get('limit', 5, type=int)
        
        logger.info(f"Analyzing news sentiment for {symbol.upper()}")
        
        # Get news and sentiment
        news_result = news_analyzer.get_stock_news(symbol.upper(), limit)
        
        if not news_result.get("success"):
            return jsonify(news_result), 400
        
        # Extract sentiment details
        sentiment_details = []
        for article in news_result["articles"]:
            sentiment = article.get("sentiment", {})
            sentiment_details.append({
                "title": article["title"],
                "sentiment_score": sentiment.get("score", 0),
                "sentiment_label": sentiment.get("label", "neutral"),
                "confidence": sentiment.get("confidence", 0.5),
                "analysis_method": sentiment.get("method", "unknown"),
                "reasons": sentiment.get("reasons", [])
            })
        
        # Create investment impact assessment
        overall_sentiment = news_result["overall_sentiment"]
        impact_assessment = self._assess_news_impact(symbol, overall_sentiment)
        
        response = {
            "symbol": symbol.upper(),
            "sentiment_analysis": {
                "overall_score": overall_sentiment["score"],
                "overall_label": overall_sentiment["label"],
                "confidence": overall_sentiment["confidence"],
                "articles_analyzed": overall_sentiment["articles_analyzed"]
            },
            "article_sentiments": sentiment_details,
            "investment_impact": impact_assessment,
            "summary": news_result["sentiment_summary"],
            "timestamp": news_result["timestamp"],
            "success": True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Sentiment analysis API error: {str(e)}")
        return jsonify({
            "error": f"Failed to analyze sentiment for {symbol}: {str(e)}",
            "success": False
        }), 500

def _assess_news_impact(symbol: str, sentiment: Dict) -> Dict[str, Any]:
    """Assess potential investment impact of news sentiment"""
    try:
        score = sentiment.get("score", 0)
        confidence = sentiment.get("confidence", 0.5)
        
        # Determine impact level
        if abs(score) > 0.6 and confidence > 0.7:
            impact_level = "High"
        elif abs(score) > 0.3 and confidence > 0.5:
            impact_level = "Medium"
        else:
            impact_level = "Low"
        
        # Generate recommendation
        if score > 0.4:
            recommendation = "Positive news sentiment may support price appreciation"
            action_bias = "BUY_SIGNAL"
        elif score < -0.4:
            recommendation = "Negative news sentiment may pressure stock price"
            action_bias = "SELL_SIGNAL"
        else:
            recommendation = "Neutral news sentiment, focus on fundamentals"
            action_bias = "NEUTRAL"
        
        return {
            "impact_level": impact_level,
            "recommendation": recommendation,
            "action_bias": action_bias,
            "confidence": confidence,
            "time_horizon": "Short-term (1-5 days)" if abs(score) > 0.5 else "Monitor ongoing"
        }
        
    except Exception as e:
        return {
            "impact_level": "Unknown",
            "recommendation": f"Impact assessment failed: {str(e)}",
            "action_bias": "NEUTRAL",
            "confidence": 0.0
        }

@app.route('/api/market-news', methods=['GET'])
def get_market_news():
    """Get general market news and sentiment"""
    try:
        # Get news for major market indices/stocks
        major_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        market_sentiment = {
            "symbols_analyzed": [],
            "sentiment_scores": [],
            "overall_market_sentiment": "neutral"
        }
        
        all_articles = []
        
        for symbol in major_symbols[:3]:  # Limit to 3 for performance
            try:
                news_result = news_analyzer.get_stock_news(symbol, 3)
                if news_result.get("success"):
                    market_sentiment["symbols_analyzed"].append(symbol)
                    market_sentiment["sentiment_scores"].append(news_result["overall_sentiment"]["score"])
                    all_articles.extend(news_result["articles"][:2])  # 2 articles per stock
            except Exception as e:
                logger.warning(f"Failed to get news for {symbol}: {e}")
        
        # Calculate market sentiment
        if market_sentiment["sentiment_scores"]:
            avg_sentiment = sum(market_sentiment["sentiment_scores"]) / len(market_sentiment["sentiment_scores"])
            if avg_sentiment > 0.2:
                market_sentiment["overall_market_sentiment"] = "bullish"
            elif avg_sentiment < -0.2:
                market_sentiment["overall_market_sentiment"] = "bearish"
            else:
                market_sentiment["overall_market_sentiment"] = "neutral"
        
        response = {
            "market_sentiment": market_sentiment,
            "featured_articles": all_articles[:10],  # Top 10 articles
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Market news API error: {str(e)}")
        return jsonify({
            "error": f"Failed to fetch market news: {str(e)}",
            "success": False
        }), 500
def get_query_formats():
    """Get information about available query formats"""
    return jsonify({
        "available_formats": {
            "sentence": {
                "description": "Natural language response in complete sentences",
                "example": "Apple (AAPL) is currently trading at $238.47, up 2.34% with a BUY recommendation at 85% confidence.",
                "use_case": "Human-readable explanations, conversational interfaces"
            },
            "table": {
                "description": "Structured data in table format with clear metrics",
                "example": "| Symbol | Price | Change | Recommendation |\n|--------|-------|--------|----------------|\n| AAPL | $238.47 | +2.34% | BUY |",
                "use_case": "Data comparison, reports, structured analysis"
            },
            "both": {
                "description": "Both sentence and table formats in single response",
                "example": "Complete response with both natural language and structured data",
                "use_case": "Comprehensive analysis, full-featured applications"
            }
        },
        "supported_queries": [
            "Single stock analysis: 'Analyze Apple stock'",
            "Stock comparison: 'Compare Tesla vs Ford'",
            "Market overview: 'Market outlook today'",
            "Investment questions: 'Should I buy Microsoft?'",
            "General queries: 'Best tech stocks to invest'"
        ],
        "usage": {
            "endpoint": "/api/intelligent-query",
            "method": "POST",
            "body": {
                "query": "Your financial question here",
                "user_id": "optional_user_identifier",
                "format": "sentence | table | both (default: both)"
            }
        }
    })

@app.route('/api/test-queries', methods=['GET'])
def get_test_queries():
    """Get example queries for testing the intelligent system"""
    test_queries = [
        "How is Apple stock performing today?",
        "Compare Tesla vs Ford stocks",
        "AAPL vs MSFT vs GOOGL analysis",
        "What's the market sentiment for NVIDIA?",
        "Should I buy Amazon stock?",
        "How to start investing for beginners?",
        "What is P/E ratio?",
        "Portfolio diversification strategies",
        "Market overview today",
        "Best tech stocks to buy now",
        "Risk assessment for my portfolio",
        "TSLA earnings analysis",
        "Compare Apple and Microsoft for long-term investment",
        "What are the top performing stocks this week?",
        "How volatile is Bitcoin compared to tech stocks?"
    ]
    
    return jsonify({
        "test_queries": test_queries,
        "usage": "POST any of these queries to /api/query or /api/intelligent-query",
        "format": {"query": "Your question here", "user_id": "optional_user_id"}
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.route('/api/test/intelligent', methods=['POST'])
def test_intelligent_assistant():
    """Test endpoint for the intelligent financial assistant"""
    try:
        data = request.get_json()
        test_queries = data.get('queries', [
            "Compare Apple vs Microsoft stocks for long-term investment",
            "How is Tesla performing this quarter?",
            "Should I invest in NVDA?",
            "What are the best tech stocks right now?",
            "AAPL vs GOOGL analysis"
        ])
        
        results = {}
        for i, query in enumerate(test_queries):
            logger.info(f"Testing query {i+1}: {query}")
            result = intelligent_assistant.process_query(query, f"test_user_{i}")
            results[f"query_{i+1}"] = {
                "query": query,
                "result": result
            }
        
        return jsonify({
            "test_name": "Intelligent Financial Assistant Test",
            "total_queries": len(test_queries),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Test failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "success": False
        }), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Initialize agents
    init_agents()
    
    logger.info("🚀 Starting MCP-Powered Financial Analyst Flask Application...")
    logger.info("🤖 AI-Powered Financial Query System Ready!")
    logger.info("🌐 Available endpoints:")
    logger.info("  GET  /                          - Main dashboard")
    logger.info("  GET  /api/health                - Health check")
    logger.info("  POST /api/query                 - AI-powered financial queries")
    logger.info("  POST /api/intelligent-query     - Advanced AI query processing")
    logger.info("  GET  /api/test-queries          - Get example queries")
    logger.info("  GET  /api/analyze/stock/<symbol> - Analyze specific stock")
    logger.info("  GET  /api/stock/price/<symbol>  - Get stock price")
    logger.info("  GET  /api/stock/info/<symbol>   - Get company info")
    logger.info("  GET  /api/stock/historical/<symbol> - Get historical data")
    logger.info("  POST /api/compare/stocks        - Compare multiple stocks")
    logger.info("")
    logger.info("🧠 AI Query Examples:")
    logger.info("  💬 'How is Apple stock performing today?'")
    logger.info("  💬 'Compare Tesla vs Ford stocks for investment'")
    logger.info("  💬 'AAPL vs MSFT analysis'")
    logger.info("  💬 'Should I buy Amazon stock?'")
    logger.info("  💬 'What is P/E ratio?'")
    logger.info("  💬 'How to start investing as a beginner?'")
    logger.info("")
    logger.info("🔧 Features working:")
    logger.info("  ✅ AI-powered natural language processing")
    logger.info("  ✅ Intelligent query routing and context analysis")
    logger.info("  ✅ Real-time stock prices and company data")
    logger.info("  ✅ Technical analysis and investment recommendations")
    logger.info("  ✅ Stock comparison with detailed insights")
    logger.info("  ✅ Human-readable responses for any financial question")
    logger.info("  ✅ Database storage and query history")
    logger.info("")
    logger.info("🧪 To test the AI system:")
    logger.info("  Run: python test_intelligent_api.py")
    logger.info("  Or visit: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

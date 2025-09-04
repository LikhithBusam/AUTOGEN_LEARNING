# """
# MCP Server implementation for Financial Data APIs
# """
# import asyncio
# import json
# from typing import Any, Dict, List, Optional
# from datetime import datetime, timedelta
# import yfinance as yf
# import requests
# from alpha_vantage.timeseries import TimeSeries
# from alpha_vantage.fundamentaldata import FundamentalData
# from config.settings import settings
# from data.database import db_manager

# class FinancialDataMCPServer:
#     """MCP Server for financial data integration"""
    
#     def __init__(self):
#         self.alpha_vantage_ts = TimeSeries(key=settings.alpha_vantage_api_key)
#         self.alpha_vantage_fd = FundamentalData(key=settings.alpha_vantage_api_key)
#         self.news_api_key = settings.news_api_key
    
#     async def get_stock_price(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
#         """Get current stock price and basic info"""
#         try:
#             # Check cache first
#             cached_data = db_manager.get_cached_stock_data(symbol, f"price_{period}")
#             if cached_data:
#                 return cached_data
            
#             # Fetch from Yahoo Finance
#             ticker = yf.Ticker(symbol)
#             data = ticker.history(period=period)
            
#             if data.empty:
#                 return {"error": f"No data found for symbol {symbol}"}
            
#             latest = data.iloc[-1]
#             result = {
#                 "symbol": symbol,
#                 "current_price": float(latest['Close']),
#                 "open": float(latest['Open']),
#                 "high": float(latest['High']),
#                 "low": float(latest['Low']),
#                 "volume": int(latest['Volume']),
#                 "change": float(latest['Close'] - data.iloc[-2]['Close']) if len(data) > 1 else 0,
#                 "change_percent": ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0,
#                 "timestamp": datetime.now().isoformat()
#             }
            
#             # Cache for 5 minutes
#             expires_at = datetime.now() + timedelta(minutes=5)
#             db_manager.cache_stock_data(symbol, f"price_{period}", result, expires_at)
            
#             return result
            
#         except Exception as e:
#             return {"error": f"Failed to fetch stock price for {symbol}: {str(e)}"}
    
#     async def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
#         """Get historical stock data"""
#         try:
#             # Check cache first (cache for 1 hour for historical data)
#             cache_key = f"historical_{symbol}_{period}_{interval}"
#             cached_data = db_manager.get_cached_stock_data(symbol, cache_key)
#             if cached_data:
#                 return cached_data
            
#             ticker = yf.Ticker(symbol)
#             data = ticker.history(period=period, interval=interval)
            
#             if data.empty:
#                 return {"error": f"No historical data found for symbol {symbol}"}
            
#             # Convert to JSON-serializable format
#             result = {
#                 "symbol": symbol,
#                 "period": period,
#                 "interval": interval,
#                 "data": []
#             }
            
#             for date, row in data.iterrows():
#                 result["data"].append({
#                     "date": date.strftime("%Y-%m-%d"),
#                     "open": float(row['Open']),
#                     "high": float(row['High']),
#                     "low": float(row['Low']),
#                     "close": float(row['Close']),
#                     "volume": int(row['Volume'])
#                 })
            
#             # Cache for 1 hour
#             expires_at = datetime.now() + timedelta(hours=1)
#             db_manager.cache_stock_data(symbol, cache_key, result, expires_at)
            
#             return result
            
#         except Exception as e:
#             return {"error": f"Failed to fetch historical data for {symbol}: {str(e)}"}
    
#     async def get_company_info(self, symbol: str) -> Dict[str, Any]:
#         """Get company fundamental information"""
#         try:
#             # Check cache first (cache for 24 hours)
#             cached_data = db_manager.get_cached_stock_data(symbol, "company_info")
#             if cached_data:
#                 return cached_data
            
#             ticker = yf.Ticker(symbol)
#             info = ticker.info
            
#             # Extract key information
#             result = {
#                 "symbol": symbol,
#                 "company_name": info.get("longName", ""),
#                 "sector": info.get("sector", ""),
#                 "industry": info.get("industry", ""),
#                 "market_cap": info.get("marketCap", 0),
#                 "pe_ratio": info.get("trailingPE", 0),
#                 "dividend_yield": info.get("dividendYield", 0),
#                 "beta": info.get("beta", 0),
#                 "52_week_high": info.get("fiftyTwoWeekHigh", 0),
#                 "52_week_low": info.get("fiftyTwoWeekLow", 0),
#                 "description": info.get("longBusinessSummary", ""),
#                 "website": info.get("website", ""),
#                 "employees": info.get("fullTimeEmployees", 0)
#             }
            
#             # Cache for 24 hours
#             expires_at = datetime.now() + timedelta(hours=24)
#             db_manager.cache_stock_data(symbol, "company_info", result, expires_at)
            
#             return result
            
#         except Exception as e:
#             return {"error": f"Failed to fetch company info for {symbol}: {str(e)}"}
    
#     async def get_financials(self, symbol: str) -> Dict[str, Any]:
#         """Get company financial statements"""
#         try:
#             # Use Alpha Vantage for fundamental data
#             income_statement, _ = self.alpha_vantage_fd.get_income_statement_annual(symbol)
#             balance_sheet, _ = self.alpha_vantage_fd.get_balance_sheet_annual(symbol)
#             cash_flow, _ = self.alpha_vantage_fd.get_cash_flow_annual(symbol)
            
#             result = {
#                 "symbol": symbol,
#                 "income_statement": income_statement,
#                 "balance_sheet": balance_sheet,
#                 "cash_flow": cash_flow,
#                 "fetched_at": datetime.now().isoformat()
#             }
            
#             return result
            
#         except Exception as e:
#             return {"error": f"Failed to fetch financials for {symbol}: {str(e)}"}
    
#     async def get_earnings_data(self, symbol: str) -> Dict[str, Any]:
#         """Get earnings data"""
#         try:
#             ticker = yf.Ticker(symbol)
            
#             # Get earnings data
#             earnings = ticker.earnings
#             quarterly_earnings = ticker.quarterly_earnings
            
#             result = {
#                 "symbol": symbol,
#                 "annual_earnings": earnings.to_dict() if not earnings.empty else {},
#                 "quarterly_earnings": quarterly_earnings.to_dict() if not quarterly_earnings.empty else {},
#                 "next_earnings_date": None,  # Would need to parse from earnings calendar
#                 "fetched_at": datetime.now().isoformat()
#             }
            
#             return result
            
#         except Exception as e:
#             return {"error": f"Failed to fetch earnings data for {symbol}: {str(e)}"}
    
#     async def search_news(self, query: str, symbols: List[str] = None, limit: int = 10) -> Dict[str, Any]:
#         """Search financial news"""
#         try:
#             news_articles = []
            
#             # Search using News API
#             if self.news_api_key:
#                 url = "https://newsapi.org/v2/everything"
#                 params = {
#                     "q": query,
#                     "apiKey": self.news_api_key,
#                     "language": "en",
#                     "sortBy": "publishedAt",
#                     "pageSize": limit
#                 }
                
#                 response = requests.get(url, params=params)
#                 if response.status_code == 200:
#                     articles = response.json().get("articles", [])
                    
#                     for article in articles:
#                         news_articles.append({
#                             "title": article.get("title", ""),
#                             "description": article.get("description", ""),
#                             "url": article.get("url", ""),
#                             "source": article.get("source", {}).get("name", ""),
#                             "published_at": article.get("publishedAt", ""),
#                             "content": article.get("content", "")
#                         })
            
#             # If symbols provided, also search Yahoo Finance news
#             if symbols:
#                 for symbol in symbols:
#                     try:
#                         ticker = yf.Ticker(symbol)
#                         ticker_news = ticker.news
                        
#                         for article in ticker_news[:5]:  # Limit to 5 per symbol
#                             news_articles.append({
#                                 "title": article.get("title", ""),
#                                 "description": "",
#                                 "url": article.get("link", ""),
#                                 "source": "Yahoo Finance",
#                                 "published_at": datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
#                                 "symbol": symbol
#                             })
#                     except:
#                         continue
            
#             return {
#                 "query": query,
#                 "symbols": symbols,
#                 "articles": news_articles[:limit],
#                 "total_found": len(news_articles),
#                 "fetched_at": datetime.now().isoformat()
#             }
            
#         except Exception as e:
#             return {"error": f"Failed to search news: {str(e)}"}

# class MCPClient:
#     """MCP Client for making requests to the financial data server"""
    
#     def __init__(self):
#         self.server = FinancialDataMCPServer()
    
#     async def call_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
#         """Call a method on the MCP server"""
#         try:
#             if method == "get_stock_price":
#                 return await self.server.get_stock_price(**params)
#             elif method == "get_historical_data":
#                 return await self.server.get_historical_data(**params)
#             elif method == "get_company_info":
#                 return await self.server.get_company_info(**params)
#             elif method == "get_financials":
#                 return await self.server.get_financials(**params)
#             elif method == "get_earnings_data":
#                 return await self.server.get_earnings_data(**params)
#             elif method == "search_news":
#                 return await self.server.search_news(**params)
#             else:
#                 return {"error": f"Unknown method: {method}"}
#         except Exception as e:
#             return {"error": f"MCP call failed: {str(e)}"}

# # Global MCP client instance
# mcp_client = MCPClient()


import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import httpx
from functools import partial

# Replace with your project's settings pattern
from config.settings import settings
# Replace with your data manager implementation  
from data.database import db_manager

class FinancialDataMCPServer:
    """Async MCP Server for financial data integration"""

    def __init__(self):
        # Initialize with basic settings, alpha_vantage will be loaded on demand
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
        self.news_api_key = settings.news_api_key
        
    def _init_alpha_vantage(self):
        """Initialize Alpha Vantage clients on demand"""
        if self.alpha_vantage_ts is None:
            try:
                from alpha_vantage.timeseries import TimeSeries
                from alpha_vantage.fundamentaldata import FundamentalData
                self.alpha_vantage_ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format='pandas')
                self.alpha_vantage_fd = FundamentalData(key=settings.alpha_vantage_api_key, output_format='pandas')
            except ImportError:
                print("Warning: alpha_vantage package not available")

    async def get_stock_price(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        try:
            # Check cache first
            cached_data = db_manager.get_cached_stock_data(symbol, f"price_{period}")
            if cached_data:
                return cached_data

            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            data = await loop.run_in_executor(None, partial(ticker.history, period=period))
            if data.empty:
                return {"error": f"No data found for symbol {symbol}"}

            latest = data.iloc[-1]
            result = {
                "symbol": symbol,
                "current_price": float(latest['Close']),
                "open": float(latest['Open']),
                "high": float(latest['High']),
                "low": float(latest['Low']),
                "volume": int(latest['Volume']),
                "change": float(latest['Close'] - data.iloc[-2]['Close']) if len(data) > 1 else 0,
                "change_percent": ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0,
                "timestamp": datetime.now().isoformat()
            }

            # Cache for 5 minutes
            expires_at = datetime.now() + timedelta(minutes=5)
            db_manager.cache_stock_data(symbol, f"price_{period}", result, expires_at)
            return result

        except Exception as e:
            return {"error": f"Failed to fetch stock price for {symbol}: {str(e)}"}

    async def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        try:
            cache_key = f"historical_{symbol}_{period}_{interval}"
            cached_data = db_manager.get_cached_stock_data(symbol, cache_key)
            if cached_data:
                return cached_data

            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            data = await loop.run_in_executor(None, partial(ticker.history, period=period, interval=interval))
            if data.empty:
                return {"error": f"No historical data found for symbol {symbol}"}

            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data": []
            }
            for date, row in data.iterrows():
                result["data"].append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                })

            expires_at = datetime.now() + timedelta(hours=1)
            db_manager.cache_stock_data(symbol, cache_key, result, expires_at)
            return result

        except Exception as e:
            return {"error": f"Failed to fetch historical data for {symbol}: {str(e)}"}

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        try:
            cached_data = db_manager.get_cached_stock_data(symbol, "company_info")
            if cached_data:
                return cached_data

            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)

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
                "employees": info.get("fullTimeEmployees", 0)
            }

            expires_at = datetime.now() + timedelta(hours=24)
            db_manager.cache_stock_data(symbol, "company_info", result, expires_at)
            return result

        except Exception as e:
            return {"error": f"Failed to fetch company info for {symbol}: {str(e)}"}

    async def get_financials(self, symbol: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            # Wrap AlphaVantage synchronous calls
            get_income = partial(self.alpha_vantage_fd.get_income_statement_annual, symbol)
            income_statement, _ = await loop.run_in_executor(None, get_income)

            get_balance = partial(self.alpha_vantage_fd.get_balance_sheet_annual, symbol)
            balance_sheet, _ = await loop.run_in_executor(None, get_balance)

            get_cash = partial(self.alpha_vantage_fd.get_cash_flow_annual, symbol)
            cash_flow, _ = await loop.run_in_executor(None, get_cash)

            result = {
                "symbol": symbol,
                "income_statement": income_statement.to_dict() if hasattr(income_statement, 'to_dict') else income_statement,
                "balance_sheet": balance_sheet.to_dict() if hasattr(balance_sheet, 'to_dict') else balance_sheet,
                "cash_flow": cash_flow.to_dict() if hasattr(cash_flow, 'to_dict') else cash_flow,
                "fetched_at": datetime.now().isoformat()
            }
            return result

        except Exception as e:
            return {"error": f"Failed to fetch financials for {symbol}: {str(e)}"}

    async def get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            earnings = await loop.run_in_executor(None, lambda: ticker.earnings)
            quarterly_earnings = await loop.run_in_executor(None, lambda: ticker.quarterly_earnings)
            result = {
                "symbol": symbol,
                "annual_earnings": earnings.to_dict() if not earnings.empty else {},
                "quarterly_earnings": quarterly_earnings.to_dict() if not quarterly_earnings.empty else {},
                "next_earnings_date": None, # Optional: Parse from ticker.calendar
                "fetched_at": datetime.now().isoformat()
            }
            return result

        except Exception as e:
            return {"error": f"Failed to fetch earnings data for {symbol}: {str(e)}"}

    async def search_news(self, query: str, symbols: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        try:
            news_articles = []

            # Async NewsAPI
            if self.news_api_key:
                async with httpx.AsyncClient() as client:
                    url = "https://newsapi.org/v2/everything"
                    params = {"q": query, "apiKey": self.news_api_key, "language": "en", "sortBy": "publishedAt", "pageSize": limit}
                    response = await client.get(url, params=params)
                    if response.status_code == 200:
                        articles = response.json().get("articles", [])
                        for article in articles:
                            news_articles.append({
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "source": article.get("source", {}).get("name", ""),
                                "published_at": article.get("publishedAt", ""),
                                "content": article.get("content", "")
                            })

            # Async Yahoo Finance news for symbols
            if symbols:
                loop = asyncio.get_event_loop()
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        ticker_news = await loop.run_in_executor(None, lambda: ticker.news)
                        for article in ticker_news[:5]:  # Limit to 5 per symbol
                            news_articles.append({
                                "title": article.get("title", ""),
                                "description": "",
                                "url": article.get("link", ""),
                                "source": "Yahoo Finance",
                                "published_at": datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                                "symbol": symbol
                            })
                    except Exception:
                        continue

            return {
                "query": query,
                "symbols": symbols,
                "articles": news_articles[:limit],
                "total_found": len(news_articles),
                "fetched_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Failed to search news: {str(e)}"}

class MCPClient:
    """Async MCP Client for making requests to the financial data server"""

    def __init__(self):
        self.server = FinancialDataMCPServer()

    async def call_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if method == "get_stock_price":
                return await self.server.get_stock_price(**params)
            elif method == "get_historical_data":
                return await self.server.get_historical_data(**params)
            elif method == "get_company_info":
                return await self.server.get_company_info(**params)
            elif method == "get_financials":
                return await self.server.get_financials(**params)
            elif method == "get_earnings_data":
                return await self.server.get_earnings_data(**params)
            elif method == "search_news":
                return await self.server.search_news(**params)
            else:
                return {"error": f"Unknown method: {method}"}
        except Exception as e:
            return {"error": f"MCP call failed: {str(e)}"}

# Global MCP client instance
mcp_client = MCPClient()

import asyncio
from typing import Any, Dict, List
from datetime import datetime, timedelta
import yfinance as yf
import httpx
from functools import partial

from config.settings import settings
from data.database import db_manager


class FinancialDataMCPServer:
    """Async MCP Server for financial data integration"""

    def __init__(self):
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
        self.news_api_key = settings.news_api_key

    def _init_alpha_vantage(self):
        """Initialize Alpha Vantage clients on demand"""
        if self.alpha_vantage_ts is None:
            try:
                from alpha_vantage.timeseries import TimeSeries
                from alpha_vantage.fundamentaldata import FundamentalData
                self.alpha_vantage_ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format="pandas")
                self.alpha_vantage_fd = FundamentalData(key=settings.alpha_vantage_api_key, output_format="pandas")
            except ImportError:
                raise RuntimeError("alpha_vantage package not installed. Run `pip install alpha_vantage`.")

    async def get_stock_price(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        try:
            cached_data = await db_manager.get_cached_stock_data(symbol, f"price_{period}")
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
                "current_price": float(latest["Close"]),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "volume": int(latest["Volume"]),
                "change": float(latest["Close"] - data.iloc[-2]["Close"]) if len(data) > 1 else 0,
                "change_percent": ((latest["Close"] - data.iloc[-2]["Close"]) / data.iloc[-2]["Close"] * 100) if len(data) > 1 else 0,
                "timestamp": datetime.now().isoformat(),
            }

            expires_at = datetime.now() + timedelta(minutes=5)
            await db_manager.cache_stock_data(symbol, f"price_{period}", result, expires_at)
            return result
        except Exception as e:
            return {"error": f"Failed to fetch stock price for {symbol}: {str(e)}"}

    async def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        try:
            cache_key = f"historical_{symbol}_{period}_{interval}"
            cached_data = await db_manager.get_cached_stock_data(symbol, cache_key)
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
                "data": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                    for date, row in data.iterrows()
                ],
            }

            expires_at = datetime.now() + timedelta(hours=1)
            await db_manager.cache_stock_data(symbol, cache_key, result, expires_at)
            return result
        except Exception as e:
            return {"error": f"Failed to fetch historical data for {symbol}: {str(e)}"}

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        try:
            cached_data = await db_manager.get_cached_stock_data(symbol, "company_info")
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
                "employees": info.get("fullTimeEmployees", 0),
            }

            expires_at = datetime.now() + timedelta(hours=24)
            await db_manager.cache_stock_data(symbol, "company_info", result, expires_at)
            return result
        except Exception as e:
            return {"error": f"Failed to fetch company info for {symbol}: {str(e)}"}

    async def get_financials(self, symbol: str) -> Dict[str, Any]:
        try:
            self._init_alpha_vantage()  # ✅ ensure initialized
            loop = asyncio.get_event_loop()

            income_statement, _ = await loop.run_in_executor(None, partial(self.alpha_vantage_fd.get_income_statement_annual, symbol))
            balance_sheet, _ = await loop.run_in_executor(None, partial(self.alpha_vantage_fd.get_balance_sheet_annual, symbol))
            cash_flow, _ = await loop.run_in_executor(None, partial(self.alpha_vantage_fd.get_cash_flow_annual, symbol))

            return {
                "symbol": symbol,
                "income_statement": income_statement.to_dict() if hasattr(income_statement, "to_dict") else income_statement,
                "balance_sheet": balance_sheet.to_dict() if hasattr(balance_sheet, "to_dict") else balance_sheet,
                "cash_flow": cash_flow.to_dict() if hasattr(cash_flow, "to_dict") else cash_flow,
                "fetched_at": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": f"Failed to fetch financials for {symbol}: {str(e)}"}

    async def get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            earnings = await loop.run_in_executor(None, lambda: ticker.earnings)
            quarterly_earnings = await loop.run_in_executor(None, lambda: ticker.quarterly_earnings)
            return {
                "symbol": symbol,
                "annual_earnings": earnings.to_dict() if not earnings.empty else {},
                "quarterly_earnings": quarterly_earnings.to_dict() if not quarterly_earnings.empty else {},
                "next_earnings_date": None,
                "fetched_at": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": f"Failed to fetch earnings data for {symbol}: {str(e)}"}

    async def search_news(self, query: str, symbols: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        try:
            news_articles = []

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
                                "content": article.get("content", ""),
                            })
                    else:
                        return {"error": f"News API request failed: {response.text}"}

            if symbols:
                loop = asyncio.get_event_loop()
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        ticker_news = await loop.run_in_executor(None, lambda: ticker.news)
                        for article in ticker_news[:5]:
                            news_articles.append({
                                "title": article.get("title", ""),
                                "description": "",
                                "url": article.get("link", ""),
                                "source": "Yahoo Finance",
                                "published_at": datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                                "symbol": symbol,
                            })
                    except Exception:
                        continue

            return {
                "query": query,
                "symbols": symbols,
                "articles": news_articles[:limit],
                "total_found": len(news_articles),
                "fetched_at": datetime.now().isoformat(),
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
            elif method == "get_company_news":
                # Map to search_news with company-specific query
                symbol = params.get("symbol", "")
                days = params.get("days", 7)
                query = f"{symbol} stock earnings financial"
                return await self.server.search_news(query=query, symbols=[symbol], limit=10)
            elif method == "get_market_news":
                # Map to search_news with market-specific query
                days = params.get("days", 3)
                query = "stock market financial economy trading"
                return await self.server.search_news(query=query, symbols=None, limit=15)
            else:
                return {"error": f"Unknown method: {method}"}
        except Exception as e:
            return {"error": f"MCP call failed: {str(e)}"}


# Global MCP client instance
mcp_client = MCPClient()

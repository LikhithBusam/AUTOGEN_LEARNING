"""
Data Analyst Agent - Retrieves and analyzes financial data via MCP
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings
from mcp.financial_data_server import mcp_client

class DataAnalystAgent:
    """Agent specialized in financial data retrieval and analysis"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["data_analyst"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["data_analyst"]["system_message"]
        )
        self.mcp_client = mcp_client
    
    async def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Comprehensive stock analysis"""
        
        try:
            # Gather all relevant data
            current_price = await self.mcp_client.call_method("get_stock_price", {"symbol": symbol})
            historical_data = await self.mcp_client.call_method("get_historical_data", {
                "symbol": symbol, 
                "period": "1y"
            })
            company_info = await self.mcp_client.call_method("get_company_info", {"symbol": symbol})
            earnings_data = await self.mcp_client.call_method("get_earnings_data", {"symbol": symbol})
            
            # Perform technical analysis
            technical_analysis = await self._perform_technical_analysis(historical_data)
            
            # Calculate key metrics
            key_metrics = await self._calculate_key_metrics(current_price, historical_data, company_info)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "historical_data": historical_data,
                "company_info": company_info,
                "earnings_data": earnings_data,
                "technical_analysis": technical_analysis,
                "key_metrics": key_metrics,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze stock {symbol}: {str(e)}"}
    
    async def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        
        try:
            stock_analyses = {}
            
            # Analyze each stock
            for symbol in symbols:
                stock_analyses[symbol] = await self.analyze_stock(symbol)
            
            # Perform comparison analysis
            comparison = await self._perform_comparison_analysis(stock_analyses)
            
            return {
                "symbols": symbols,
                "individual_analyses": stock_analyses,
                "comparison": comparison,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to compare stocks: {str(e)}"}
    
    async def get_earnings_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyze earnings data for a stock"""
        
        try:
            earnings_data = await self.mcp_client.call_method("get_earnings_data", {"symbol": symbol})
            company_info = await self.mcp_client.call_method("get_company_info", {"symbol": symbol})
            
            # Analyze earnings trends
            earnings_analysis = await self._analyze_earnings_trends(earnings_data)
            
            return {
                "symbol": symbol,
                "earnings_data": earnings_data,
                "company_info": company_info,
                "earnings_analysis": earnings_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze earnings for {symbol}: {str(e)}"}
    
    async def get_portfolio_analysis(self, portfolio: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze a portfolio of stocks"""
        
        try:
            portfolio_data = {}
            total_value = 0
            total_cost = 0
            
            # Analyze each holding
            for symbol, holding in portfolio.items():
                shares = holding.get("shares", 0)
                avg_cost = holding.get("avg_cost", 0)
                
                current_price_data = await self.mcp_client.call_method("get_stock_price", {"symbol": symbol})
                current_price = current_price_data.get("current_price", 0)
                
                position_value = shares * current_price
                position_cost = shares * avg_cost
                gain_loss = position_value - position_cost
                gain_loss_percent = (gain_loss / position_cost * 100) if position_cost > 0 else 0
                
                portfolio_data[symbol] = {
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "position_value": position_value,
                    "position_cost": position_cost,
                    "gain_loss": gain_loss,
                    "gain_loss_percent": gain_loss_percent,
                    "weight": 0  # Will calculate after getting total value
                }
                
                total_value += position_value
                total_cost += position_cost
            
            # Calculate weights and portfolio metrics
            for symbol in portfolio_data:
                portfolio_data[symbol]["weight"] = (portfolio_data[symbol]["position_value"] / total_value * 100) if total_value > 0 else 0
            
            portfolio_metrics = {
                "total_value": total_value,
                "total_cost": total_cost,
                "total_gain_loss": total_value - total_cost,
                "total_gain_loss_percent": ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
                "number_of_positions": len(portfolio),
                "largest_position": max(portfolio_data.items(), key=lambda x: x[1]["position_value"])[0] if portfolio_data else None,
                "best_performer": max(portfolio_data.items(), key=lambda x: x[1]["gain_loss_percent"])[0] if portfolio_data else None,
                "worst_performer": min(portfolio_data.items(), key=lambda x: x[1]["gain_loss_percent"])[0] if portfolio_data else None
            }
            
            return {
                "portfolio": portfolio_data,
                "metrics": portfolio_metrics,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze portfolio: {str(e)}"}
    
    async def _perform_technical_analysis(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis on historical data"""
        
        try:
            if "error" in historical_data or not historical_data.get("data"):
                return {"error": "No historical data available for technical analysis"}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(historical_data["data"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Calculate technical indicators
            analysis = {}
            
            # Simple Moving Averages
            analysis["sma_20"] = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            analysis["sma_50"] = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            analysis["sma_200"] = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
            
            # Current price relative to SMAs
            current_price = df['close'].iloc[-1]
            analysis["price_vs_sma_20"] = ((current_price / analysis["sma_20"]) - 1) * 100 if analysis["sma_20"] else None
            analysis["price_vs_sma_50"] = ((current_price / analysis["sma_50"]) - 1) * 100 if analysis["sma_50"] else None
            
            # Volatility (20-day)
            analysis["volatility_20d"] = df['close'].pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100 if len(df) >= 20 else None
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            analysis["rsi"] = 100 - (100 / (1 + rs)).iloc[-1] if len(df) >= 14 else None
            
            # Support and Resistance levels
            analysis["support_level"] = df['low'].rolling(window=20).min().iloc[-1] if len(df) >= 20 else None
            analysis["resistance_level"] = df['high'].rolling(window=20).max().iloc[-1] if len(df) >= 20 else None
            
            # Trend analysis
            if len(df) >= 50:
                recent_trend = df['close'].iloc[-20:].mean() / df['close'].iloc[-50:-30].mean() - 1
                analysis["trend_strength"] = recent_trend * 100
                analysis["trend_direction"] = "bullish" if recent_trend > 0.02 else "bearish" if recent_trend < -0.02 else "sideways"
            
            return analysis
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    async def _calculate_key_metrics(self, current_price: Dict[str, Any], historical_data: Dict[str, Any], company_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key financial metrics"""
        
        try:
            metrics = {}
            
            # Price metrics
            if "current_price" in current_price:
                metrics["current_price"] = current_price["current_price"]
                metrics["daily_change"] = current_price.get("change", 0)
                metrics["daily_change_percent"] = current_price.get("change_percent", 0)
            
            # Valuation metrics from company info
            metrics["market_cap"] = company_info.get("market_cap", 0)
            metrics["pe_ratio"] = company_info.get("pe_ratio", 0)
            metrics["dividend_yield"] = company_info.get("dividend_yield", 0)
            metrics["beta"] = company_info.get("beta", 0)
            
            # Historical performance
            if not historical_data.get("error") and historical_data.get("data"):
                df = pd.DataFrame(historical_data["data"])
                if len(df) > 0:
                    # Performance calculations
                    latest_price = df['close'].iloc[-1]
                    
                    if len(df) >= 30:
                        month_ago_price = df['close'].iloc[-30]
                        metrics["1m_return"] = ((latest_price / month_ago_price) - 1) * 100
                    
                    if len(df) >= 90:
                        quarter_ago_price = df['close'].iloc[-90]
                        metrics["3m_return"] = ((latest_price / quarter_ago_price) - 1) * 100
                    
                    if len(df) >= 252:
                        year_ago_price = df['close'].iloc[-252]
                        metrics["1y_return"] = ((latest_price / year_ago_price) - 1) * 100
                    
                    # Volatility
                    returns = pd.Series(df['close']).pct_change().dropna()
                    metrics["annual_volatility"] = returns.std() * np.sqrt(252) * 100
                    
                    # 52-week high/low
                    metrics["52w_high"] = df['high'].max()
                    metrics["52w_low"] = df['low'].min()
                    metrics["price_vs_52w_high"] = ((latest_price / metrics["52w_high"]) - 1) * 100
                    metrics["price_vs_52w_low"] = ((latest_price / metrics["52w_low"]) - 1) * 100
            
            return metrics
            
        except Exception as e:
            return {"error": f"Key metrics calculation failed: {str(e)}"}
    
    async def _perform_comparison_analysis(self, stock_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        
        try:
            comparison = {
                "performance_comparison": {},
                "valuation_comparison": {},
                "risk_comparison": {},
                "summary": {}
            }
            
            # Extract key metrics for comparison
            symbols = list(stock_analyses.keys())
            
            for metric in ["daily_change_percent", "1m_return", "3m_return", "1y_return"]:
                comparison["performance_comparison"][metric] = {}
                for symbol in symbols:
                    metrics = stock_analyses[symbol].get("key_metrics", {})
                    comparison["performance_comparison"][metric][symbol] = metrics.get(metric, 0)
            
            for metric in ["pe_ratio", "market_cap", "dividend_yield"]:
                comparison["valuation_comparison"][metric] = {}
                for symbol in symbols:
                    metrics = stock_analyses[symbol].get("key_metrics", {})
                    comparison["valuation_comparison"][metric][symbol] = metrics.get(metric, 0)
            
            for metric in ["beta", "annual_volatility"]:
                comparison["risk_comparison"][metric] = {}
                for symbol in symbols:
                    metrics = stock_analyses[symbol].get("key_metrics", {})
                    comparison["risk_comparison"][metric][symbol] = metrics.get(metric, 0)
            
            # Generate summary
            comparison["summary"]["best_performer_ytd"] = self._find_best_performer(comparison["performance_comparison"].get("1y_return", {}))
            comparison["summary"]["lowest_volatility"] = self._find_lowest_value(comparison["risk_comparison"].get("annual_volatility", {}))
            comparison["summary"]["highest_dividend"] = self._find_highest_value(comparison["valuation_comparison"].get("dividend_yield", {}))
            
            return comparison
            
        except Exception as e:
            return {"error": f"Comparison analysis failed: {str(e)}"}
    
    async def _analyze_earnings_trends(self, earnings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze earnings trends"""
        
        try:
            analysis = {}
            
            quarterly_earnings = earnings_data.get("quarterly_earnings", {})
            annual_earnings = earnings_data.get("annual_earnings", {})
            
            if quarterly_earnings:
                # Analyze quarterly trends
                quarters = list(quarterly_earnings.keys())
                if len(quarters) >= 4:
                    # Calculate quarter-over-quarter growth
                    latest_quarter = quarters[0]
                    previous_quarter = quarters[1]
                    
                    if quarterly_earnings[latest_quarter] and quarterly_earnings[previous_quarter]:
                        qoq_growth = ((quarterly_earnings[latest_quarter] / quarterly_earnings[previous_quarter]) - 1) * 100
                        analysis["qoq_growth"] = qoq_growth
            
            if annual_earnings:
                # Analyze annual trends
                years = list(annual_earnings.keys())
                if len(years) >= 2:
                    latest_year = years[0]
                    previous_year = years[1]
                    
                    if annual_earnings[latest_year] and annual_earnings[previous_year]:
                        yoy_growth = ((annual_earnings[latest_year] / annual_earnings[previous_year]) - 1) * 100
                        analysis["yoy_growth"] = yoy_growth
                        
                    # Calculate earnings stability (coefficient of variation)
                    earnings_values = [v for v in annual_earnings.values() if v is not None]
                    if len(earnings_values) >= 3:
                        mean_earnings = np.mean(earnings_values)
                        std_earnings = np.std(earnings_values)
                        analysis["earnings_stability"] = (std_earnings / mean_earnings) if mean_earnings != 0 else None
            
            return analysis
            
        except Exception as e:
            return {"error": f"Earnings analysis failed: {str(e)}"}
    
    def _find_best_performer(self, performance_dict: Dict[str, float]) -> str:
        """Find the best performing stock"""
        if not performance_dict:
            return None
        return max(performance_dict.items(), key=lambda x: x[1] or 0)[0]
    
    def _find_lowest_value(self, values_dict: Dict[str, float]) -> str:
        """Find the stock with lowest value for a metric"""
        if not values_dict:
            return None
        return min(values_dict.items(), key=lambda x: x[1] or float('inf'))[0]
    
    def _find_highest_value(self, values_dict: Dict[str, float]) -> str:
        """Find the stock with highest value for a metric"""
        if not values_dict:
            return None
        return max(values_dict.items(), key=lambda x: x[1] or 0)[0]

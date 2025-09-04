"""
Visualization Agent - Creates charts and graphs for financial data
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings

class VisualizationAgent:
    """Agent specialized in creating financial visualizations and charts"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["visualization"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["visualization"]["system_message"]
        )
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create charts directory
        self.charts_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
    
    async def create_stock_price_chart(self, historical_data: Dict[str, Any], symbol: str, chart_type: str = "candlestick") -> Dict[str, Any]:
        """Create stock price chart with various technical indicators"""
        
        try:
            if "error" in historical_data or not historical_data.get("data"):
                return {"error": "No historical data available for chart creation"}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data["data"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            if chart_type.lower() == "candlestick":
                chart_path = await self._create_candlestick_chart(df, symbol)
            elif chart_type.lower() == "line":
                chart_path = await self._create_line_chart(df, symbol)
            elif chart_type.lower() == "ohlc":
                chart_path = await self._create_ohlc_chart(df, symbol)
            else:
                chart_path = await self._create_candlestick_chart(df, symbol)  # Default
            
            return {
                "success": True,
                "chart_path": chart_path,
                "chart_type": chart_type,
                "symbol": symbol,
                "data_points": len(df),
                "date_range": {
                    "start": df.index.min().strftime("%Y-%m-%d"),
                    "end": df.index.max().strftime("%Y-%m-%d")
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to create stock price chart: {str(e)}"}
    
    async def create_portfolio_allocation_chart(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create portfolio allocation pie chart and bar chart"""
        
        try:
            portfolio = portfolio_data.get("portfolio", {})
            
            if not portfolio:
                return {"error": "No portfolio data available for visualization"}
            
            # Create both pie chart and bar chart
            pie_chart_path = await self._create_portfolio_pie_chart(portfolio)
            bar_chart_path = await self._create_portfolio_bar_chart(portfolio)
            
            return {
                "success": True,
                "pie_chart": pie_chart_path,
                "bar_chart": bar_chart_path,
                "portfolio_size": len(portfolio),
                "total_value": sum(pos.get("position_value", 0) for pos in portfolio.values())
            }
            
        except Exception as e:
            return {"error": f"Failed to create portfolio charts: {str(e)}"}
    
    async def create_performance_comparison_chart(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance comparison chart for multiple stocks"""
        
        try:
            individual_analyses = comparison_data.get("individual_analyses", {})
            
            if not individual_analyses:
                return {"error": "No comparison data available for visualization"}
            
            # Extract performance metrics
            performance_data = self._extract_performance_metrics(individual_analyses)
            
            # Create comparison charts
            returns_chart = await self._create_returns_comparison_chart(performance_data)
            metrics_chart = await self._create_metrics_comparison_chart(performance_data)
            
            return {
                "success": True,
                "returns_chart": returns_chart,
                "metrics_chart": metrics_chart,
                "stocks_compared": len(individual_analyses)
            }
            
        except Exception as e:
            return {"error": f"Failed to create comparison charts: {str(e)}"}
    
    async def create_sentiment_visualization(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create sentiment analysis visualizations"""
        
        try:
            # Create sentiment score gauge
            gauge_chart = await self._create_sentiment_gauge(sentiment_data)
            
            # Create sector sentiment chart if available
            sector_chart = None
            if sentiment_data.get("sector_sentiment"):
                sector_chart = await self._create_sector_sentiment_chart(sentiment_data)
            
            return {
                "success": True,
                "sentiment_gauge": gauge_chart,
                "sector_chart": sector_chart,
                "sentiment_score": sentiment_data.get("market_sentiment", {}).get("score", 0)
            }
            
        except Exception as e:
            return {"error": f"Failed to create sentiment visualizations: {str(e)}"}
    
    async def create_technical_analysis_chart(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive technical analysis chart"""
        
        try:
            historical_data = analysis_data.get("historical_data", {})
            technical_analysis = analysis_data.get("technical_analysis", {})
            symbol = analysis_data.get("symbol", "Unknown")
            
            if "error" in historical_data or not historical_data.get("data"):
                return {"error": "No historical data available for technical analysis chart"}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data["data"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Create technical analysis chart with indicators
            chart_path = await self._create_technical_indicators_chart(df, technical_analysis, symbol)
            
            return {
                "success": True,
                "chart_path": chart_path,
                "symbol": symbol,
                "indicators_included": list(technical_analysis.keys())
            }
            
        except Exception as e:
            return {"error": f"Failed to create technical analysis chart: {str(e)}"}
    
    async def create_risk_return_scatter(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk-return scatter plot for portfolio positions"""
        
        try:
            portfolio = portfolio_data.get("portfolio", {})
            
            if not portfolio:
                return {"error": "No portfolio data available for risk-return analysis"}
            
            # Extract risk-return data
            risk_return_data = self._extract_risk_return_data(portfolio)
            
            # Create scatter plot
            chart_path = await self._create_risk_return_chart(risk_return_data)
            
            return {
                "success": True,
                "chart_path": chart_path,
                "positions_analyzed": len(risk_return_data)
            }
            
        except Exception as e:
            return {"error": f"Failed to create risk-return chart: {str(e)}"}
    
    async def _create_candlestick_chart(self, df: pd.DataFrame, symbol: str) -> str:
        """Create interactive candlestick chart with volume"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Stock Price', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Add moving averages if enough data
        if len(df) >= 20:
            ma20 = df['close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if len(df) >= 50:
            ma50 = df['close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Add volume
        colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            yaxis_title='Stock Price ($)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        # Save chart
        filename = f"{symbol}_candlestick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_line_chart(self, df: pd.DataFrame, symbol: str) -> str:
        """Create simple line chart for stock price"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot closing price
        ax.plot(df.index, df['close'], linewidth=2, label=f'{symbol} Close Price')
        
        # Add moving averages
        if len(df) >= 20:
            ma20 = df['close'].rolling(window=20).mean()
            ax.plot(df.index, ma20, alpha=0.8, label='20-day MA')
        
        if len(df) >= 50:
            ma50 = df['close'].rolling(window=50).mean()
            ax.plot(df.index, ma50, alpha=0.8, label='50-day MA')
        
        ax.set_title(f'{symbol} Stock Price Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        filename = f"{symbol}_line_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        chart_path = os.path.join(self.charts_dir, filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    async def _create_ohlc_chart(self, df: pd.DataFrame, symbol: str) -> str:
        """Create OHLC bar chart"""
        
        fig = go.Figure(data=go.Ohlc(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        ))
        
        fig.update_layout(
            title=f'{symbol} OHLC Chart',
            yaxis_title='Stock Price ($)',
            xaxis_title='Date',
            height=600
        )
        
        # Save chart
        filename = f"{symbol}_ohlc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_portfolio_pie_chart(self, portfolio: Dict[str, Any]) -> str:
        """Create portfolio allocation pie chart"""
        
        # Extract data for pie chart
        symbols = list(portfolio.keys())
        weights = [pos.get("weight", 0) for pos in portfolio.values()]
        values = [pos.get("position_value", 0) for pos in portfolio.values()]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: $%{value:,.2f}<br>' +
                         'Weight: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title='Portfolio Allocation',
            showlegend=True,
            height=600
        )
        
        # Save chart
        filename = f"portfolio_pie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_portfolio_bar_chart(self, portfolio: Dict[str, Any]) -> str:
        """Create portfolio position values bar chart"""
        
        # Extract data
        symbols = list(portfolio.keys())
        values = [pos.get("position_value", 0) for pos in portfolio.values()]
        gains_losses = [pos.get("gain_loss_percent", 0) for pos in portfolio.values()]
        
        # Create color map based on gains/losses
        colors = ['green' if gl > 0 else 'red' if gl < 0 else 'gray' for gl in gains_losses]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=values,
                marker_color=colors,
                text=[f'${v:,.0f}<br>{gl:+.1f}%' for v, gl in zip(values, gains_losses)],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Portfolio Position Values',
            xaxis_title='Symbol',
            yaxis_title='Position Value ($)',
            height=500
        )
        
        # Save chart
        filename = f"portfolio_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    def _extract_performance_metrics(self, individual_analyses: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract performance metrics for comparison"""
        
        performance_data = {}
        
        for symbol, analysis in individual_analyses.items():
            if "error" not in analysis:
                key_metrics = analysis.get("key_metrics", {})
                performance_data[symbol] = {
                    "daily_change": key_metrics.get("daily_change_percent", 0),
                    "1m_return": key_metrics.get("1m_return", 0),
                    "3m_return": key_metrics.get("3m_return", 0),
                    "1y_return": key_metrics.get("1y_return", 0),
                    "volatility": key_metrics.get("annual_volatility", 0),
                    "pe_ratio": key_metrics.get("pe_ratio", 0),
                    "market_cap": key_metrics.get("market_cap", 0)
                }
        
        return performance_data
    
    async def _create_returns_comparison_chart(self, performance_data: Dict[str, Dict[str, float]]) -> str:
        """Create returns comparison chart"""
        
        symbols = list(performance_data.keys())
        periods = ["daily_change", "1m_return", "3m_return", "1y_return"]
        period_labels = ["Daily", "1 Month", "3 Months", "1 Year"]
        
        fig = go.Figure()
        
        for symbol in symbols:
            returns = [performance_data[symbol].get(period, 0) for period in periods]
            fig.add_trace(go.Bar(
                name=symbol,
                x=period_labels,
                y=returns,
                text=[f'{r:+.2f}%' for r in returns],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Performance Comparison',
            xaxis_title='Time Period',
            yaxis_title='Return (%)',
            barmode='group',
            height=500
        )
        
        # Save chart
        filename = f"returns_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_metrics_comparison_chart(self, performance_data: Dict[str, Dict[str, float]]) -> str:
        """Create valuation metrics comparison chart"""
        
        symbols = list(performance_data.keys())
        pe_ratios = [performance_data[symbol].get("pe_ratio", 0) for symbol in symbols]
        volatilities = [performance_data[symbol].get("volatility", 0) for symbol in symbols]
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('P/E Ratios', 'Volatility'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # P/E ratios
        fig.add_trace(
            go.Bar(x=symbols, y=pe_ratios, name='P/E Ratio', marker_color='blue'),
            row=1, col=1
        )
        
        # Volatility
        fig.add_trace(
            go.Bar(x=symbols, y=volatilities, name='Volatility (%)', marker_color='red'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Valuation & Risk Metrics Comparison',
            height=500,
            showlegend=False
        )
        
        # Save chart
        filename = f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_sentiment_gauge(self, sentiment_data: Dict[str, Any]) -> str:
        """Create sentiment gauge chart"""
        
        market_sentiment = sentiment_data.get("market_sentiment", {})
        sentiment_score = market_sentiment.get("score", 0)
        sentiment_label = market_sentiment.get("label", "neutral")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Market Sentiment: {sentiment_label.title()}"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=400)
        
        # Save chart
        filename = f"sentiment_gauge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_sector_sentiment_chart(self, sentiment_data: Dict[str, Any]) -> str:
        """Create sector sentiment comparison chart"""
        
        sector_sentiment = sentiment_data.get("sector_sentiment", {})
        
        if not sector_sentiment:
            return None
        
        sectors = list(sector_sentiment.keys())
        scores = list(sector_sentiment.values())
        
        # Create color map
        colors = ['green' if score > 0.1 else 'red' if score < -0.1 else 'gray' for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sectors,
                y=scores,
                marker_color=colors,
                text=[f'{score:+.2f}' for score in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Sector Sentiment Analysis',
            xaxis_title='Sector',
            yaxis_title='Sentiment Score',
            height=400
        )
        
        # Save chart
        filename = f"sector_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    async def _create_technical_indicators_chart(self, df: pd.DataFrame, technical_analysis: Dict[str, Any], symbol: str) -> str:
        """Create comprehensive technical analysis chart"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f'{symbol} Price & Moving Averages', 'RSI', 'Volume'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if len(df) >= 20:
            ma20 = df['close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if len(df) >= 50:
            ma50 = df['close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Support and resistance levels
        support = technical_analysis.get("support_level")
        resistance = technical_analysis.get("resistance_level")
        
        if support:
            fig.add_hline(y=support, line_dash="dash", line_color="green", 
                         annotation_text="Support", row=1, col=1)
        
        if resistance:
            fig.add_hline(y=resistance, line_dash="dash", line_color="red", 
                         annotation_text="Resistance", row=1, col=1)
        
        # RSI
        rsi_value = technical_analysis.get("rsi", 50)
        if len(df) >= 14:
            # Calculate RSI for visualization
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=900,
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        # Save chart
        filename = f"{symbol}_technical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path
    
    def _extract_risk_return_data(self, portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract risk-return data for scatter plot"""
        
        risk_return_data = []
        
        for symbol, position in portfolio.items():
            # Use gain/loss percent as return proxy
            return_value = position.get("gain_loss_percent", 0)
            
            # Estimate risk based on position size and volatility (simplified)
            weight = position.get("weight", 0)
            estimated_volatility = 20  # Default volatility estimate
            
            risk_return_data.append({
                "symbol": symbol,
                "return": return_value,
                "risk": estimated_volatility,
                "weight": weight,
                "value": position.get("position_value", 0)
            })
        
        return risk_return_data
    
    async def _create_risk_return_chart(self, risk_return_data: List[Dict[str, Any]]) -> str:
        """Create risk-return scatter plot"""
        
        symbols = [item["symbol"] for item in risk_return_data]
        returns = [item["return"] for item in risk_return_data]
        risks = [item["risk"] for item in risk_return_data]
        weights = [item["weight"] for item in risk_return_data]
        
        fig = go.Figure(data=go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=symbols,
            textposition="top center",
            marker=dict(
                size=[w/2 for w in weights],  # Size proportional to weight
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return (%)"),
                line=dict(width=2, color='black')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Risk: %{x:.1f}%<br>' +
                         'Return: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Portfolio Risk-Return Analysis',
            xaxis_title='Risk (Estimated Volatility %)',
            yaxis_title='Return (%)',
            height=600,
            showlegend=False
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=20, line_dash="dash", line_color="gray")
        
        # Save chart
        filename = f"risk_return_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        chart_path = os.path.join(self.charts_dir, filename)
        fig.write_html(chart_path)
        
        return chart_path

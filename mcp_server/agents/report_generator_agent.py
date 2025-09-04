"""
Report Generator Agent - Creates comprehensive financial reports
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings

# Optional weasyprint import with fallback
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except Exception as e:
    print(f"Warning: weasyprint not available ({type(e).__name__}: {e}). PDF generation will be limited.")
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None

class ReportGeneratorAgent:
    """Agent specialized in generating comprehensive financial reports"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["report_generator"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["report_generator"]["system_message"]
        )
        
        # Setup Jinja2 environment for templates
        template_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "templates")
        os.makedirs(template_dir, exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    async def generate_stock_analysis_report(self, analysis_data: Dict[str, Any], format: str = "html") -> Dict[str, Any]:
        """Generate a comprehensive stock analysis report"""
        
        try:
            # Prepare report data
            report_data = self._prepare_stock_report_data(analysis_data)
            
            # Generate report content
            if format.lower() == "pdf":
                report_content = await self._generate_pdf_report(report_data, "stock_analysis")
                filename = f"stock_analysis_{report_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                report_content = await self._generate_html_report(report_data, "stock_analysis")
                filename = f"stock_analysis_{report_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Save report
            report_path = await self._save_report(report_content, filename, format)
            
            return {
                "success": True,
                "report_path": report_path,
                "filename": filename,
                "format": format,
                "symbol": report_data["symbol"],
                "generated_at": datetime.now().isoformat(),
                "summary": self._generate_report_summary(report_data)
            }
            
        except Exception as e:
            return {"error": f"Failed to generate stock analysis report: {str(e)}"}
    
    async def generate_portfolio_report(self, portfolio_data: Dict[str, Any], format: str = "html") -> Dict[str, Any]:
        """Generate a comprehensive portfolio report"""
        
        try:
            # Prepare portfolio report data
            report_data = self._prepare_portfolio_report_data(portfolio_data)
            
            # Generate report content
            if format.lower() == "pdf":
                report_content = await self._generate_pdf_report(report_data, "portfolio_analysis")
                filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                report_content = await self._generate_html_report(report_data, "portfolio_analysis")
                filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Save report
            report_path = await self._save_report(report_content, filename, format)
            
            return {
                "success": True,
                "report_path": report_path,
                "filename": filename,
                "format": format,
                "portfolio_value": report_data["metrics"]["total_value"],
                "positions": len(report_data["portfolio"]),
                "generated_at": datetime.now().isoformat(),
                "summary": self._generate_portfolio_summary(report_data)
            }
            
        except Exception as e:
            return {"error": f"Failed to generate portfolio report: {str(e)}"}
    
    async def generate_market_sentiment_report(self, sentiment_data: Dict[str, Any], format: str = "html") -> Dict[str, Any]:
        """Generate a market sentiment analysis report"""
        
        try:
            # Prepare sentiment report data
            report_data = self._prepare_sentiment_report_data(sentiment_data)
            
            # Generate report content
            if format.lower() == "pdf":
                report_content = await self._generate_pdf_report(report_data, "sentiment_analysis")
                filename = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                report_content = await self._generate_html_report(report_data, "sentiment_analysis")
                filename = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Save report
            report_path = await self._save_report(report_content, filename, format)
            
            return {
                "success": True,
                "report_path": report_path,
                "filename": filename,
                "format": format,
                "sentiment_score": report_data["market_sentiment"]["score"],
                "articles_analyzed": report_data["article_count"],
                "generated_at": datetime.now().isoformat(),
                "summary": self._generate_sentiment_summary(report_data)
            }
            
        except Exception as e:
            return {"error": f"Failed to generate sentiment report: {str(e)}"}
    
    async def generate_comparison_report(self, comparison_data: Dict[str, Any], format: str = "html") -> Dict[str, Any]:
        """Generate a stock comparison report"""
        
        try:
            # Prepare comparison report data
            report_data = self._prepare_comparison_report_data(comparison_data)
            
            # Generate report content
            if format.lower() == "pdf":
                report_content = await self._generate_pdf_report(report_data, "stock_comparison")
                filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                report_content = await self._generate_html_report(report_data, "stock_comparison")
                filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Save report
            report_path = await self._save_report(report_content, filename, format)
            
            return {
                "success": True,
                "report_path": report_path,
                "filename": filename,
                "format": format,
                "symbols": report_data["symbols"],
                "generated_at": datetime.now().isoformat(),
                "summary": self._generate_comparison_summary(report_data)
            }
            
        except Exception as e:
            return {"error": f"Failed to generate comparison report: {str(e)}"}
    
    def _prepare_stock_report_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for stock analysis report"""
        
        symbol = analysis_data.get("symbol", "Unknown")
        current_price = analysis_data.get("current_price", {})
        key_metrics = analysis_data.get("key_metrics", {})
        technical_analysis = analysis_data.get("technical_analysis", {})
        company_info = analysis_data.get("company_info", {})
        
        return {
            "symbol": symbol,
            "company_name": company_info.get("company_name", symbol),
            "current_price": current_price.get("current_price", 0),
            "daily_change": current_price.get("change_percent", 0),
            "key_metrics": key_metrics,
            "technical_analysis": technical_analysis,
            "company_info": company_info,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyst_summary": self._generate_analyst_summary(analysis_data),
            "risk_assessment": self._assess_investment_risk(analysis_data),
            "price_targets": self._calculate_price_targets(analysis_data)
        }
    
    def _prepare_portfolio_report_data(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for portfolio report"""
        
        portfolio = portfolio_data.get("portfolio", {})
        metrics = portfolio_data.get("metrics", {})
        
        # Calculate additional portfolio insights
        portfolio_insights = self._calculate_portfolio_insights(portfolio_data)
        
        return {
            "portfolio": portfolio,
            "metrics": metrics,
            "insights": portfolio_insights,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diversification_analysis": self._analyze_diversification(portfolio),
            "risk_analysis": self._analyze_portfolio_risk(portfolio),
            "recommendations": self._generate_portfolio_recommendations(portfolio_data)
        }
    
    def _prepare_sentiment_report_data(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for sentiment report"""
        
        return {
            "market_sentiment": sentiment_data.get("market_sentiment", {}),
            "themes": sentiment_data.get("themes", {}),
            "sector_sentiment": sentiment_data.get("sector_sentiment", {}),
            "article_count": sentiment_data.get("article_count", 0),
            "top_concerns": sentiment_data.get("top_concerns", []),
            "top_opportunities": sentiment_data.get("top_opportunities", []),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment_interpretation": self._interpret_sentiment_score(sentiment_data.get("market_sentiment", {})),
            "market_outlook": self._generate_market_outlook(sentiment_data)
        }
    
    def _prepare_comparison_report_data(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for comparison report"""
        
        symbols = comparison_data.get("symbols", [])
        individual_analyses = comparison_data.get("individual_analyses", {})
        comparison = comparison_data.get("comparison", {})
        
        return {
            "symbols": symbols,
            "individual_analyses": individual_analyses,
            "comparison": comparison,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "winner_analysis": self._determine_comparison_winners(comparison),
            "investment_recommendations": self._generate_comparison_recommendations(comparison_data)
        }
    
    async def _generate_html_report(self, report_data: Dict[str, Any], template_name: str) -> str:
        """Generate HTML report using Jinja2 template"""
        
        try:
            template = self.jinja_env.get_template(f"{template_name}.html")
            html_content = template.render(**report_data)
            return html_content
        except Exception as e:
            # Fallback to basic HTML structure
            return self._generate_basic_html_report(report_data, template_name)
    
    async def _generate_pdf_report(self, report_data: Dict[str, Any], template_name: str) -> bytes:
        """Generate PDF report from HTML template"""
        
        try:
            if not WEASYPRINT_AVAILABLE:
                # Fallback: return HTML as bytes with warning
                html_content = await self._generate_html_report(report_data, template_name)
                warning_html = f"""
                <div style="background: #fff3cd; padding: 10px; border: 1px solid #ffeaa7; margin: 10px; border-radius: 5px;">
                    <strong>Note:</strong> PDF generation not available. WeasyPrint requires additional system libraries on Windows.
                    <br>Please use HTML format or install WeasyPrint dependencies.
                </div>
                {html_content}
                """
                return warning_html.encode('utf-8')
            
            # First generate HTML
            html_content = await self._generate_html_report(report_data, template_name)
            
            # Convert to PDF
            html = HTML(string=html_content)
            pdf_bytes = html.write_pdf()
            
            return pdf_bytes
        except Exception as e:
            # Fallback to basic PDF generation
            return self._generate_basic_pdf_report(report_data, template_name)
    
    async def _save_report(self, report_content: Any, filename: str, format: str) -> str:
        """Save report to file system"""
        
        # Create reports directory
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "generated")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, filename)
        
        try:
            if format.lower() == "pdf":
                with open(report_path, "wb") as f:
                    f.write(report_content)
            else:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
            
            return report_path
        except Exception as e:
            return f"Error saving report: {str(e)}"
    
    def _generate_analyst_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Generate analyst summary from stock analysis data"""
        
        key_metrics = analysis_data.get("key_metrics", {})
        technical_analysis = analysis_data.get("technical_analysis", {})
        symbol = analysis_data.get("symbol", "Unknown")
        
        # Analyze key indicators
        pe_ratio = key_metrics.get("pe_ratio", 0)
        daily_change = key_metrics.get("daily_change_percent", 0)
        rsi = technical_analysis.get("rsi", 50)
        
        summary_points = []
        
        # Price action analysis
        if daily_change > 2:
            summary_points.append(f"{symbol} showed strong positive momentum with a {daily_change:.1f}% gain.")
        elif daily_change < -2:
            summary_points.append(f"{symbol} experienced selling pressure with a {daily_change:.1f}% decline.")
        else:
            summary_points.append(f"{symbol} traded relatively flat with minimal price movement.")
        
        # Valuation analysis
        if pe_ratio > 0:
            if pe_ratio > 25:
                summary_points.append(f"The stock appears expensive with a P/E ratio of {pe_ratio:.1f}.")
            elif pe_ratio < 15:
                summary_points.append(f"The stock appears undervalued with a P/E ratio of {pe_ratio:.1f}.")
            else:
                summary_points.append(f"The stock is fairly valued with a P/E ratio of {pe_ratio:.1f}.")
        
        # Technical analysis
        if rsi > 70:
            summary_points.append("Technical indicators suggest the stock may be overbought.")
        elif rsi < 30:
            summary_points.append("Technical indicators suggest the stock may be oversold.")
        else:
            summary_points.append("Technical indicators show neutral momentum.")
        
        return " ".join(summary_points)
    
    def _assess_investment_risk(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess investment risk based on analysis data"""
        
        key_metrics = analysis_data.get("key_metrics", {})
        technical_analysis = analysis_data.get("technical_analysis", {})
        
        risk_factors = []
        risk_score = 0  # 0-10 scale
        
        # Volatility risk
        volatility = key_metrics.get("annual_volatility", 0)
        if volatility > 30:
            risk_factors.append("High volatility")
            risk_score += 2
        elif volatility > 20:
            risk_factors.append("Moderate volatility")
            risk_score += 1
        
        # Beta risk
        beta = key_metrics.get("beta", 1)
        if beta > 1.5:
            risk_factors.append("High market sensitivity")
            risk_score += 1
        
        # Technical risk
        rsi = technical_analysis.get("rsi", 50)
        if rsi > 80 or rsi < 20:
            risk_factors.append("Extreme technical conditions")
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_level)
        }
    
    def _calculate_price_targets(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate price targets based on technical analysis"""
        
        current_price = analysis_data.get("current_price", {}).get("current_price", 0)
        technical_analysis = analysis_data.get("technical_analysis", {})
        
        if current_price == 0:
            return {}
        
        # Simple price target calculation
        support = technical_analysis.get("support_level", current_price * 0.9)
        resistance = technical_analysis.get("resistance_level", current_price * 1.1)
        
        return {
            "support": support,
            "resistance": resistance,
            "target_price": (resistance + current_price) / 2,
            "stop_loss": support * 0.95
        }
    
    def _calculate_portfolio_insights(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio insights"""
        
        portfolio = portfolio_data.get("portfolio", {})
        metrics = portfolio_data.get("metrics", {})
        
        insights = {}
        
        # Performance insights
        best_performer = metrics.get("best_performer", "N/A")
        worst_performer = metrics.get("worst_performer", "N/A")
        
        insights["top_performer"] = best_performer
        insights["bottom_performer"] = worst_performer
        
        # Concentration analysis
        if portfolio:
            weights = [pos.get("weight", 0) for pos in portfolio.values()]
            max_weight = max(weights) if weights else 0
            
            if max_weight > 30:
                insights["concentration_risk"] = "High - Consider diversifying"
            elif max_weight > 20:
                insights["concentration_risk"] = "Moderate - Monitor closely"
            else:
                insights["concentration_risk"] = "Low - Well diversified"
        
        return insights
    
    def _analyze_diversification(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        
        # Simple diversification analysis
        num_positions = len(portfolio)
        
        if num_positions < 5:
            diversification_score = "Poor"
            recommendation = "Consider adding more positions"
        elif num_positions < 10:
            diversification_score = "Fair"
            recommendation = "Good start, consider expanding"
        elif num_positions < 20:
            diversification_score = "Good"
            recommendation = "Well diversified portfolio"
        else:
            diversification_score = "Excellent"
            recommendation = "Highly diversified, monitor for over-diversification"
        
        return {
            "score": diversification_score,
            "position_count": num_positions,
            "recommendation": recommendation
        }
    
    def _analyze_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio risk"""
        
        if not portfolio:
            return {"risk_level": "Unknown", "analysis": "No portfolio data available"}
        
        # Calculate weighted average of position risks
        total_weight = 0
        weighted_volatility = 0
        
        for symbol, position in portfolio.items():
            weight = position.get("weight", 0) / 100
            # Use a default volatility estimate if not available
            volatility = 20  # Default 20% annual volatility
            
            weighted_volatility += weight * volatility
            total_weight += weight
        
        if total_weight > 0:
            portfolio_volatility = weighted_volatility / total_weight
        else:
            portfolio_volatility = 20
        
        if portfolio_volatility > 25:
            risk_level = "High"
        elif portfolio_volatility > 15:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            "risk_level": risk_level,
            "estimated_volatility": portfolio_volatility,
            "analysis": f"Portfolio estimated annual volatility: {portfolio_volatility:.1f}%"
        }
    
    def _generate_portfolio_recommendations(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Generate portfolio recommendations"""
        
        recommendations = []
        portfolio = portfolio_data.get("portfolio", {})
        metrics = portfolio_data.get("metrics", {})
        
        # Performance-based recommendations
        total_return = metrics.get("total_gain_loss_percent", 0)
        if total_return < -10:
            recommendations.append("Consider reviewing positions with significant losses")
        elif total_return > 20:
            recommendations.append("Strong performance - consider taking some profits")
        
        # Diversification recommendations
        num_positions = len(portfolio)
        if num_positions < 5:
            recommendations.append("Increase diversification by adding more positions")
        
        # Position size recommendations
        for symbol, position in portfolio.items():
            weight = position.get("weight", 0)
            if weight > 25:
                recommendations.append(f"Consider reducing position size in {symbol} (currently {weight:.1f}%)")
        
        return recommendations
    
    def _interpret_sentiment_score(self, sentiment: Dict[str, Any]) -> str:
        """Interpret sentiment score for human readers"""
        
        score = sentiment.get("score", 0)
        label = sentiment.get("label", "neutral")
        confidence = sentiment.get("confidence", 0)
        
        if label == "positive":
            if score > 0.5:
                return f"Very positive sentiment with high confidence ({confidence:.1%})"
            else:
                return f"Moderately positive sentiment with {confidence:.1%} confidence"
        elif label == "negative":
            if score < -0.5:
                return f"Very negative sentiment with high confidence ({confidence:.1%})"
            else:
                return f"Moderately negative sentiment with {confidence:.1%} confidence"
        else:
            return f"Neutral sentiment - no clear directional bias"
    
    def _generate_market_outlook(self, sentiment_data: Dict[str, Any]) -> str:
        """Generate market outlook based on sentiment data"""
        
        market_sentiment = sentiment_data.get("market_sentiment", {})
        themes = sentiment_data.get("themes", {})
        concerns = sentiment_data.get("top_concerns", [])
        opportunities = sentiment_data.get("top_opportunities", [])
        
        outlook_points = []
        
        # Sentiment-based outlook
        sentiment_score = market_sentiment.get("score", 0)
        if sentiment_score > 0.2:
            outlook_points.append("Market sentiment is generally optimistic")
        elif sentiment_score < -0.2:
            outlook_points.append("Market sentiment shows caution and concern")
        else:
            outlook_points.append("Market sentiment is mixed with no clear direction")
        
        # Concerns
        if concerns:
            outlook_points.append(f"Key concerns include: {', '.join(concerns[:3])}")
        
        # Opportunities
        if opportunities:
            outlook_points.append(f"Potential opportunities: {', '.join(opportunities[:3])}")
        
        return ". ".join(outlook_points) + "."
    
    def _determine_comparison_winners(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """Determine winners in different categories"""
        
        winners = {}
        
        # Performance winner
        performance = comparison.get("performance_comparison", {})
        ytd_returns = performance.get("1y_return", {})
        if ytd_returns:
            winners["best_performance"] = max(ytd_returns.items(), key=lambda x: x[1] or 0)[0]
        
        # Value winner (lowest P/E)
        valuation = comparison.get("valuation_comparison", {})
        pe_ratios = valuation.get("pe_ratio", {})
        if pe_ratios:
            valid_pe = {k: v for k, v in pe_ratios.items() if v and v > 0}
            if valid_pe:
                winners["best_value"] = min(valid_pe.items(), key=lambda x: x[1])[0]
        
        # Income winner (highest dividend)
        dividend_yields = valuation.get("dividend_yield", {})
        if dividend_yields:
            winners["best_income"] = max(dividend_yields.items(), key=lambda x: x[1] or 0)[0]
        
        return winners
    
    def _generate_comparison_recommendations(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate investment recommendations from comparison"""
        
        recommendations = []
        comparison = comparison_data.get("comparison", {})
        winners = self._determine_comparison_winners(comparison)
        
        if "best_performance" in winners:
            recommendations.append(f"For growth: Consider {winners['best_performance']} (best 1-year performance)")
        
        if "best_value" in winners:
            recommendations.append(f"For value: Consider {winners['best_value']} (lowest P/E ratio)")
        
        if "best_income" in winners:
            recommendations.append(f"For income: Consider {winners['best_income']} (highest dividend yield)")
        
        return recommendations
    
    def _generate_report_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate a brief summary of the stock report"""
        
        symbol = report_data.get("symbol", "Unknown")
        current_price = report_data.get("current_price", 0)
        daily_change = report_data.get("daily_change", 0)
        
        return f"{symbol} analysis: Current price ${current_price:.2f} ({daily_change:+.2f}%)"
    
    def _generate_portfolio_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate a brief summary of the portfolio report"""
        
        metrics = report_data.get("metrics", {})
        total_value = metrics.get("total_value", 0)
        total_return = metrics.get("total_gain_loss_percent", 0)
        positions = len(report_data.get("portfolio", {}))
        
        return f"Portfolio: ${total_value:,.2f} value, {total_return:+.1f}% return, {positions} positions"
    
    def _generate_sentiment_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate a brief summary of the sentiment report"""
        
        sentiment = report_data.get("market_sentiment", {})
        score = sentiment.get("score", 0)
        label = sentiment.get("label", "neutral")
        articles = report_data.get("article_count", 0)
        
        return f"Market sentiment: {label.title()} (score: {score:.2f}) from {articles} articles"
    
    def _generate_comparison_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate a brief summary of the comparison report"""
        
        symbols = report_data.get("symbols", [])
        return f"Comparison analysis of {len(symbols)} stocks: {', '.join(symbols)}"
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get investment recommendation based on risk level"""
        
        if risk_level == "High":
            return "High risk investment - suitable for aggressive investors only"
        elif risk_level == "Moderate":
            return "Moderate risk investment - suitable for balanced portfolios"
        else:
            return "Low risk investment - suitable for conservative investors"
    
    def _generate_basic_html_report(self, report_data: Dict[str, Any], template_name: str) -> str:
        """Generate basic HTML report as fallback"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; }}
                .content {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Financial Analysis Report</h1>
                <p>Generated on: {report_data.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
            </div>
            <div class="content">
                <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_basic_pdf_report(self, report_data: Dict[str, Any], template_name: str) -> bytes:
        """Generate basic PDF report as fallback"""
        
        html_content = self._generate_basic_html_report(report_data, template_name)
        
        try:
            html = HTML(string=html_content)
            return html.write_pdf()
        except Exception:
            # Return HTML content as bytes if PDF generation fails
            return html_content.encode('utf-8')
    
    def _create_default_templates(self):
        """Create default Jinja2 templates"""
        
        template_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "templates")
        
        # Stock analysis template
        stock_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Analysis Report - {{ symbol }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px 15px 10px 0; }
        .positive { color: green; }
        .negative { color: red; }
        .neutral { color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ company_name }} ({{ symbol }})</h1>
        <h2>${{ "%.2f"|format(current_price) }} 
        <span class="{% if daily_change > 0 %}positive{% elif daily_change < 0 %}negative{% else %}neutral{% endif %}">
            ({{ "%+.2f"|format(daily_change) }}%)
        </span></h2>
        <p>Generated: {{ generated_at }}</p>
    </div>
    
    <div class="section">
        <h3>Key Metrics</h3>
        {% for key, value in key_metrics.items() %}
            <div class="metric"><strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}</div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h3>Analyst Summary</h3>
        <p>{{ analyst_summary }}</p>
    </div>
    
    <div class="section">
        <h3>Risk Assessment</h3>
        <p><strong>Risk Level:</strong> {{ risk_assessment.risk_level }}</p>
        <p><strong>Risk Factors:</strong> {{ risk_assessment.risk_factors | join(', ') }}</p>
        <p><strong>Recommendation:</strong> {{ risk_assessment.recommendation }}</p>
    </div>
</body>
</html>
        """
        
        with open(os.path.join(template_dir, "stock_analysis.html"), "w") as f:
            f.write(stock_template)
        
        # Portfolio template
        portfolio_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #27ae60; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Portfolio Analysis Report</h1>
        <h2>${{ "{:,.2f}"|format(metrics.total_value) }} 
        <span class="{% if metrics.total_gain_loss_percent > 0 %}positive{% else %}negative{% endif %}">
            ({{ "%+.2f"|format(metrics.total_gain_loss_percent) }}%)
        </span></h2>
        <p>Generated: {{ generated_at }}</p>
    </div>
    
    <div class="section">
        <h3>Portfolio Holdings</h3>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Shares</th>
                    <th>Current Price</th>
                    <th>Value</th>
                    <th>Gain/Loss</th>
                    <th>Weight</th>
                </tr>
            </thead>
            <tbody>
                {% for symbol, position in portfolio.items() %}
                <tr>
                    <td>{{ symbol }}</td>
                    <td>{{ position.shares }}</td>
                    <td>${{ "%.2f"|format(position.current_price) }}</td>
                    <td>${{ "{:,.2f}"|format(position.position_value) }}</td>
                    <td class="{% if position.gain_loss_percent > 0 %}positive{% else %}negative{% endif %}">
                        {{ "%+.2f"|format(position.gain_loss_percent) }}%
                    </td>
                    <td>{{ "%.1f"|format(position.weight) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h3>Recommendations</h3>
        <ul>
            {% for rec in recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
        """
        
        with open(os.path.join(template_dir, "portfolio_analysis.html"), "w") as f:
            f.write(portfolio_template)

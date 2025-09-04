"""
Recommendation Agent - Provides investment recommendations and portfolio optimization
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings

class RecommendationAgent:
    """Agent specialized in investment recommendations and portfolio optimization"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["recommendation"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["recommendation"]["system_message"]
        )
        
        # Risk tolerance levels
        self.risk_profiles = {
            "conservative": {"target_return": 0.06, "max_volatility": 0.10, "max_position": 0.15},
            "moderate": {"target_return": 0.10, "max_volatility": 0.15, "max_position": 0.20},
            "aggressive": {"target_return": 0.15, "max_volatility": 0.25, "max_position": 0.30}
        }
    
    async def analyze_stock_recommendation(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate buy/sell/hold recommendation for a stock"""
        
        try:
            symbol = analysis_data.get("symbol", "Unknown")
            key_metrics = analysis_data.get("key_metrics", {})
            technical_analysis = analysis_data.get("technical_analysis", {})
            company_info = analysis_data.get("company_info", {})
            
            # Calculate recommendation score
            recommendation_score = await self._calculate_recommendation_score(
                key_metrics, technical_analysis, company_info
            )
            
            # Determine recommendation
            recommendation = self._determine_recommendation(recommendation_score)
            
            # Calculate price targets
            price_targets = self._calculate_enhanced_price_targets(analysis_data)
            
            # Generate reasoning
            reasoning = self._generate_recommendation_reasoning(
                recommendation_score, key_metrics, technical_analysis
            )
            
            return {
                "symbol": symbol,
                "recommendation": recommendation["action"],
                "confidence": recommendation["confidence"],
                "recommendation_score": recommendation_score,
                "price_targets": price_targets,
                "reasoning": reasoning,
                "risk_assessment": self._assess_stock_risk(analysis_data),
                "time_horizon": recommendation["time_horizon"],
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate stock recommendation: {str(e)}"}
    
    async def optimize_portfolio(self, portfolio_data: Dict[str, Any], risk_profile: str = "moderate") -> Dict[str, Any]:
        """Optimize portfolio allocation using Modern Portfolio Theory"""
        
        try:
            portfolio = portfolio_data.get("portfolio", {})
            
            if len(portfolio) < 2:
                return {"error": "Need at least 2 positions for portfolio optimization"}
            
            # Get risk profile parameters
            profile_params = self.risk_profiles.get(risk_profile, self.risk_profiles["moderate"])
            
            # Calculate optimal weights
            optimization_result = await self._optimize_portfolio_weights(portfolio, profile_params)
            
            # Generate rebalancing recommendations
            rebalancing_recommendations = self._generate_rebalancing_recommendations(
                portfolio, optimization_result["optimal_weights"]
            )
            
            # Calculate expected portfolio metrics
            expected_metrics = self._calculate_expected_portfolio_metrics(
                optimization_result, profile_params
            )
            
            return {
                "current_portfolio": portfolio,
                "risk_profile": risk_profile,
                "optimization_result": optimization_result,
                "rebalancing_recommendations": rebalancing_recommendations,
                "expected_metrics": expected_metrics,
                "implementation_notes": self._generate_implementation_notes(rebalancing_recommendations),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to optimize portfolio: {str(e)}"}
    
    async def generate_diversification_recommendations(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations to improve portfolio diversification"""
        
        try:
            portfolio = portfolio_data.get("portfolio", {})
            
            # Analyze current diversification
            diversification_analysis = self._analyze_portfolio_diversification(portfolio)
            
            # Generate diversification recommendations
            recommendations = self._generate_diversification_suggestions(diversification_analysis)
            
            # Suggest new positions
            new_position_suggestions = self._suggest_new_positions(portfolio, diversification_analysis)
            
            return {
                "current_diversification": diversification_analysis,
                "diversification_score": self._calculate_diversification_score(diversification_analysis),
                "recommendations": recommendations,
                "new_position_suggestions": new_position_suggestions,
                "implementation_priority": self._prioritize_diversification_actions(recommendations),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate diversification recommendations: {str(e)}"}
    
    async def compare_investment_options(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple investment options and rank them"""
        
        try:
            individual_analyses = comparison_data.get("individual_analyses", {})
            
            if len(individual_analyses) < 2:
                return {"error": "Need at least 2 stocks for comparison"}
            
            # Score each stock
            stock_scores = {}
            for symbol, analysis in individual_analyses.items():
                if "error" not in analysis:
                    stock_scores[symbol] = await self._calculate_investment_score(analysis)
            
            # Rank stocks
            ranked_stocks = self._rank_investment_options(stock_scores)
            
            # Generate comparison insights
            comparison_insights = self._generate_comparison_insights(individual_analyses, ranked_stocks)
            
            # Generate allocation recommendations
            allocation_recommendations = self._generate_allocation_recommendations(ranked_stocks)
            
            return {
                "stock_scores": stock_scores,
                "ranked_stocks": ranked_stocks,
                "comparison_insights": comparison_insights,
                "allocation_recommendations": allocation_recommendations,
                "top_pick": ranked_stocks[0] if ranked_stocks else None,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to compare investment options: {str(e)}"}
    
    async def generate_market_timing_signals(self, sentiment_data: Dict[str, Any], market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate market timing signals based on sentiment and technical indicators"""
        
        try:
            # Analyze sentiment signals
            sentiment_signals = self._analyze_sentiment_signals(sentiment_data)
            
            # Generate timing recommendations
            timing_signals = self._generate_timing_signals(sentiment_signals, market_data)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(timing_signals)
            
            return {
                "sentiment_signals": sentiment_signals,
                "timing_signals": timing_signals,
                "signal_strength": signal_strength,
                "market_outlook": self._generate_market_outlook_recommendation(timing_signals),
                "action_recommendations": self._generate_timing_action_recommendations(timing_signals),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate market timing signals: {str(e)}"}
    
    async def _calculate_recommendation_score(self, key_metrics: Dict[str, Any], 
                                           technical_analysis: Dict[str, Any], 
                                           company_info: Dict[str, Any]) -> float:
        """Calculate overall recommendation score (-100 to +100)"""
        
        score = 0
        total_weight = 0
        
        # Valuation score (30% weight)
        valuation_score = self._calculate_valuation_score(key_metrics, company_info)
        score += valuation_score * 0.30
        total_weight += 0.30
        
        # Technical score (25% weight)
        technical_score = self._calculate_technical_score(technical_analysis)
        score += technical_score * 0.25
        total_weight += 0.25
        
        # Performance score (25% weight)
        performance_score = self._calculate_performance_score(key_metrics)
        score += performance_score * 0.25
        total_weight += 0.25
        
        # Quality score (20% weight)
        quality_score = self._calculate_quality_score(company_info, key_metrics)
        score += quality_score * 0.20
        total_weight += 0.20
        
        return score / total_weight if total_weight > 0 else 0
    
    def _calculate_valuation_score(self, key_metrics: Dict[str, Any], company_info: Dict[str, Any]) -> float:
        """Calculate valuation score based on P/E, P/B, etc."""
        
        score = 0
        
        # P/E ratio analysis
        pe_ratio = key_metrics.get("pe_ratio", 0)
        if pe_ratio > 0:
            if pe_ratio < 15:
                score += 20  # Undervalued
            elif pe_ratio < 25:
                score += 10  # Fair value
            elif pe_ratio < 35:
                score -= 10  # Overvalued
            else:
                score -= 20  # Very overvalued
        
        # Price vs 52-week high/low
        price_vs_52w_high = key_metrics.get("price_vs_52w_high", 0)
        if price_vs_52w_high < -20:
            score += 15  # Near 52-week low, potential value
        elif price_vs_52w_high > -5:
            score -= 10  # Near 52-week high, expensive
        
        # Market cap consideration
        market_cap = company_info.get("market_cap", 0)
        if market_cap > 10e9:  # Large cap
            score += 5  # Stability bonus
        elif market_cap < 1e9:  # Small cap
            score -= 5  # Higher risk
        
        return max(-50, min(50, score))
    
    def _calculate_technical_score(self, technical_analysis: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        
        score = 0
        
        # RSI analysis
        rsi = technical_analysis.get("rsi", 50)
        if rsi < 30:
            score += 20  # Oversold, potential buy
        elif rsi > 70:
            score -= 20  # Overbought, potential sell
        elif 40 <= rsi <= 60:
            score += 5   # Neutral zone
        
        # Moving average analysis
        price_vs_sma_20 = technical_analysis.get("price_vs_sma_20", 0)
        price_vs_sma_50 = technical_analysis.get("price_vs_sma_50", 0)
        
        if price_vs_sma_20 > 2 and price_vs_sma_50 > 2:
            score += 15  # Above both MAs, bullish
        elif price_vs_sma_20 < -2 and price_vs_sma_50 < -2:
            score -= 15  # Below both MAs, bearish
        
        # Trend analysis
        trend_direction = technical_analysis.get("trend_direction", "sideways")
        if trend_direction == "bullish":
            score += 10
        elif trend_direction == "bearish":
            score -= 10
        
        # Support/resistance levels
        support_level = technical_analysis.get("support_level")
        resistance_level = technical_analysis.get("resistance_level")
        if support_level and resistance_level:
            # Near support = buy opportunity
            # Near resistance = sell opportunity
            score += 5  # Basic bonus for having clear levels
        
        return max(-50, min(50, score))
    
    def _calculate_performance_score(self, key_metrics: Dict[str, Any]) -> float:
        """Calculate performance-based score"""
        
        score = 0
        
        # Recent performance
        daily_change = key_metrics.get("daily_change_percent", 0)
        month_return = key_metrics.get("1m_return", 0)
        quarter_return = key_metrics.get("3m_return", 0)
        year_return = key_metrics.get("1y_return", 0)
        
        # Year-to-date performance (highest weight)
        if year_return > 20:
            score += 20
        elif year_return > 10:
            score += 10
        elif year_return > 0:
            score += 5
        elif year_return < -20:
            score -= 20
        elif year_return < -10:
            score -= 10
        
        # Quarter performance
        if quarter_return > 10:
            score += 10
        elif quarter_return < -10:
            score -= 10
        
        # Volatility consideration
        volatility = key_metrics.get("annual_volatility", 20)
        if volatility < 15:
            score += 5  # Low volatility bonus
        elif volatility > 30:
            score -= 5  # High volatility penalty
        
        return max(-50, min(50, score))
    
    def _calculate_quality_score(self, company_info: Dict[str, Any], key_metrics: Dict[str, Any]) -> float:
        """Calculate quality score based on financial strength"""
        
        score = 0
        
        # Dividend yield
        dividend_yield = key_metrics.get("dividend_yield", 0)
        if dividend_yield > 4:
            score += 15  # High dividend
        elif dividend_yield > 2:
            score += 10  # Moderate dividend
        elif dividend_yield > 0:
            score += 5   # Some dividend
        
        # Beta (stability measure)
        beta = key_metrics.get("beta", 1)
        if 0.8 <= beta <= 1.2:
            score += 10  # Stable beta
        elif beta > 1.5:
            score -= 5   # High beta (volatile)
        
        # Market cap stability
        market_cap = company_info.get("market_cap", 0)
        if market_cap > 50e9:  # Mega cap
            score += 10  # Stability
        elif market_cap > 10e9:  # Large cap
            score += 5
        
        return max(-50, min(50, score))
    
    def _determine_recommendation(self, score: float) -> Dict[str, Any]:
        """Determine buy/sell/hold recommendation from score"""
        
        if score >= 30:
            return {
                "action": "Strong Buy",
                "confidence": min(100, abs(score)),
                "time_horizon": "3-12 months"
            }
        elif score >= 15:
            return {
                "action": "Buy",
                "confidence": min(100, abs(score)),
                "time_horizon": "6-12 months"
            }
        elif score >= 5:
            return {
                "action": "Weak Buy",
                "confidence": min(100, abs(score)),
                "time_horizon": "6-18 months"
            }
        elif score >= -5:
            return {
                "action": "Hold",
                "confidence": 50 + abs(score) * 2,
                "time_horizon": "Monitor"
            }
        elif score >= -15:
            return {
                "action": "Weak Sell",
                "confidence": min(100, abs(score)),
                "time_horizon": "3-6 months"
            }
        elif score >= -30:
            return {
                "action": "Sell",
                "confidence": min(100, abs(score)),
                "time_horizon": "1-3 months"
            }
        else:
            return {
                "action": "Strong Sell",
                "confidence": min(100, abs(score)),
                "time_horizon": "Immediate"
            }
    
    def _calculate_enhanced_price_targets(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate enhanced price targets using multiple methods"""
        
        current_price_data = analysis_data.get("current_price", {})
        current_price = current_price_data.get("current_price", 0)
        technical_analysis = analysis_data.get("technical_analysis", {})
        key_metrics = analysis_data.get("key_metrics", {})
        
        if current_price == 0:
            return {}
        
        targets = {}
        
        # Technical targets
        support = technical_analysis.get("support_level", current_price * 0.9)
        resistance = technical_analysis.get("resistance_level", current_price * 1.1)
        
        targets["support"] = support
        targets["resistance"] = resistance
        targets["stop_loss"] = support * 0.95
        
        # Performance-based targets
        year_return = key_metrics.get("1y_return", 0)
        if year_return > 0:
            # If positive momentum, extend target
            targets["12_month_target"] = current_price * (1 + max(0.15, year_return / 100))
        else:
            # Conservative target if negative momentum
            targets["12_month_target"] = current_price * 1.05
        
        # Volatility-based targets
        volatility = key_metrics.get("annual_volatility", 20) / 100
        targets["high_target"] = current_price * (1 + volatility)
        targets["low_target"] = current_price * (1 - volatility)
        
        return targets
    
    def _generate_recommendation_reasoning(self, score: float, key_metrics: Dict[str, Any], 
                                         technical_analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for recommendation"""
        
        reasoning = []
        
        # Performance reasoning
        year_return = key_metrics.get("1y_return", 0)
        if year_return > 15:
            reasoning.append(f"Strong 1-year performance of {year_return:.1f}%")
        elif year_return < -15:
            reasoning.append(f"Poor 1-year performance of {year_return:.1f}%")
        
        # Valuation reasoning
        pe_ratio = key_metrics.get("pe_ratio", 0)
        if pe_ratio > 0:
            if pe_ratio < 15:
                reasoning.append(f"Attractive valuation with P/E of {pe_ratio:.1f}")
            elif pe_ratio > 30:
                reasoning.append(f"High valuation with P/E of {pe_ratio:.1f}")
        
        # Technical reasoning
        rsi = technical_analysis.get("rsi", 50)
        if rsi < 30:
            reasoning.append("Oversold conditions suggest potential upside")
        elif rsi > 70:
            reasoning.append("Overbought conditions suggest potential downside")
        
        # Trend reasoning
        trend_direction = technical_analysis.get("trend_direction", "sideways")
        if trend_direction == "bullish":
            reasoning.append("Positive technical trend supports upside potential")
        elif trend_direction == "bearish":
            reasoning.append("Negative technical trend suggests caution")
        
        # Position vs moving averages
        price_vs_sma_20 = technical_analysis.get("price_vs_sma_20", 0)
        if price_vs_sma_20 > 5:
            reasoning.append("Price momentum above key moving averages")
        elif price_vs_sma_20 < -5:
            reasoning.append("Price below key moving averages indicates weakness")
        
        return reasoning if reasoning else ["Mixed signals across multiple factors"]
    
    def _assess_stock_risk(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess investment risk for the stock"""
        
        key_metrics = analysis_data.get("key_metrics", {})
        
        risk_factors = []
        risk_score = 0
        
        # Volatility risk
        volatility = key_metrics.get("annual_volatility", 20)
        if volatility > 30:
            risk_factors.append("High volatility")
            risk_score += 3
        elif volatility > 20:
            risk_factors.append("Moderate volatility")
            risk_score += 1
        
        # Beta risk
        beta = key_metrics.get("beta", 1)
        if beta > 1.5:
            risk_factors.append("High market sensitivity")
            risk_score += 2
        elif beta < 0.5:
            risk_factors.append("Low market correlation")
            risk_score += 1
        
        # Performance risk
        year_return = key_metrics.get("1y_return", 0)
        if year_return < -30:
            risk_factors.append("Significant recent losses")
            risk_score += 3
        elif year_return < -15:
            risk_factors.append("Recent underperformance")
            risk_score += 1
        
        # Valuation risk
        pe_ratio = key_metrics.get("pe_ratio", 0)
        if pe_ratio > 40:
            risk_factors.append("High valuation multiple")
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "High"
        elif risk_score >= 3:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "suitability": self._determine_suitability(risk_level)
        }
    
    def _determine_suitability(self, risk_level: str) -> str:
        """Determine investor suitability based on risk level"""
        
        if risk_level == "High":
            return "Suitable for aggressive investors with high risk tolerance"
        elif risk_level == "Moderate":
            return "Suitable for moderate investors seeking balanced risk-reward"
        else:
            return "Suitable for conservative investors seeking capital preservation"
    
    async def _optimize_portfolio_weights(self, portfolio: Dict[str, Any], 
                                        profile_params: Dict[str, float]) -> Dict[str, Any]:
        """Optimize portfolio weights using simplified Modern Portfolio Theory"""
        
        try:
            symbols = list(portfolio.keys())
            n_assets = len(symbols)
            
            # Get current weights
            current_weights = np.array([portfolio[symbol].get("weight", 0) / 100 for symbol in symbols])
            
            # Simulate expected returns and covariance (simplified)
            expected_returns = self._estimate_expected_returns(portfolio)
            cov_matrix = self._estimate_covariance_matrix(portfolio)
            
            # Optimization constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds for each weight
            max_weight = profile_params["max_position"]
            bounds = [(0.0, max_weight) for _ in range(n_assets)]
            
            # Objective function (minimize negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return -portfolio_return
                return -(portfolio_return / portfolio_vol)  # Negative Sharpe ratio
            
            # Initial guess (equal weights)
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate optimized portfolio metrics
                opt_return = np.dot(optimal_weights, expected_returns)
                opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                opt_sharpe = opt_return / opt_vol if opt_vol > 0 else 0
                
                return {
                    "success": True,
                    "optimal_weights": {symbol: weight for symbol, weight in zip(symbols, optimal_weights)},
                    "current_weights": {symbol: weight for symbol, weight in zip(symbols, current_weights)},
                    "expected_return": opt_return,
                    "expected_volatility": opt_vol,
                    "sharpe_ratio": opt_sharpe,
                    "optimization_message": "Portfolio optimization completed successfully"
                }
            else:
                # Fallback to equal weighting
                equal_weights = np.array([1.0 / n_assets] * n_assets)
                return {
                    "success": False,
                    "optimal_weights": {symbol: weight for symbol, weight in zip(symbols, equal_weights)},
                    "current_weights": {symbol: weight for symbol, weight in zip(symbols, current_weights)},
                    "optimization_message": "Optimization failed, using equal weights"
                }
                
        except Exception as e:
            # Return current weights as fallback
            symbols = list(portfolio.keys())
            current_weights = {symbol: portfolio[symbol].get("weight", 0) / 100 for symbol in symbols}
            
            return {
                "success": False,
                "optimal_weights": current_weights,
                "current_weights": current_weights,
                "optimization_message": f"Optimization error: {str(e)}"
            }
    
    def _estimate_expected_returns(self, portfolio: Dict[str, Any]) -> np.ndarray:
        """Estimate expected returns for portfolio assets"""
        
        returns = []
        for symbol, position in portfolio.items():
            # Use 1-year return as proxy for expected return, or default
            year_return = position.get("gain_loss_percent", 0) / 100
            expected_return = max(-0.5, min(0.5, year_return))  # Cap between -50% and +50%
            returns.append(expected_return)
        
        return np.array(returns)
    
    def _estimate_covariance_matrix(self, portfolio: Dict[str, Any]) -> np.ndarray:
        """Estimate covariance matrix for portfolio assets"""
        
        n_assets = len(portfolio)
        
        # Simplified covariance matrix (diagonal with correlation assumptions)
        cov_matrix = np.zeros((n_assets, n_assets))
        
        # Default volatilities and correlations
        default_vol = 0.20  # 20% annual volatility
        default_corr = 0.3  # 30% correlation between assets
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    cov_matrix[i, j] = default_vol ** 2  # Variance on diagonal
                else:
                    cov_matrix[i, j] = default_corr * default_vol * default_vol  # Covariance
        
        return cov_matrix
    
    def _generate_rebalancing_recommendations(self, portfolio: Dict[str, Any], 
                                            optimal_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate specific rebalancing recommendations"""
        
        recommendations = []
        
        for symbol in portfolio.keys():
            current_weight = portfolio[symbol].get("weight", 0) / 100
            optimal_weight = optimal_weights.get(symbol, 0)
            weight_diff = optimal_weight - current_weight
            
            if abs(weight_diff) > 0.02:  # Only recommend if difference > 2%
                if weight_diff > 0:
                    action = "Buy"
                    amount = weight_diff
                else:
                    action = "Sell"
                    amount = abs(weight_diff)
                
                recommendations.append({
                    "symbol": symbol,
                    "action": action,
                    "current_weight": current_weight * 100,
                    "target_weight": optimal_weight * 100,
                    "weight_change": amount * 100,
                    "priority": "High" if abs(weight_diff) > 0.05 else "Medium"
                })
        
        return sorted(recommendations, key=lambda x: abs(x["weight_change"]), reverse=True)
    
    def _calculate_expected_portfolio_metrics(self, optimization_result: Dict[str, Any], 
                                            profile_params: Dict[str, float]) -> Dict[str, Any]:
        """Calculate expected portfolio performance metrics"""
        
        expected_return = optimization_result.get("expected_return", 0)
        expected_volatility = optimization_result.get("expected_volatility", 0)
        sharpe_ratio = optimization_result.get("sharpe_ratio", 0)
        
        # Compare to target profile
        target_return = profile_params["target_return"]
        max_volatility = profile_params["max_volatility"]
        
        return {
            "expected_annual_return": expected_return * 100,
            "expected_annual_volatility": expected_volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "target_return": target_return * 100,
            "max_volatility": max_volatility * 100,
            "return_vs_target": (expected_return - target_return) * 100,
            "volatility_vs_max": (expected_volatility - max_volatility) * 100,
            "risk_adjusted_score": sharpe_ratio * 100
        }
    
    def _generate_implementation_notes(self, rebalancing_recommendations: List[Dict[str, Any]]) -> List[str]:
        """Generate implementation notes for rebalancing"""
        
        notes = []
        
        if not rebalancing_recommendations:
            notes.append("Portfolio is well-balanced, no rebalancing needed")
            return notes
        
        high_priority = [r for r in rebalancing_recommendations if r["priority"] == "High"]
        if high_priority:
            notes.append(f"High priority: Rebalance {len(high_priority)} positions")
        
        total_trades = len(rebalancing_recommendations)
        notes.append(f"Total recommended trades: {total_trades}")
        
        # Transaction cost consideration
        if total_trades > 5:
            notes.append("Consider transaction costs when implementing multiple trades")
        
        # Timing consideration
        notes.append("Consider implementing rebalancing gradually over time")
        
        return notes
    
    def _analyze_portfolio_diversification(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current portfolio diversification"""
        
        analysis = {}
        
        # Number of positions
        num_positions = len(portfolio)
        analysis["position_count"] = num_positions
        
        # Concentration analysis
        weights = [pos.get("weight", 0) for pos in portfolio.values()]
        max_weight = max(weights) if weights else 0
        top3_weight = sum(sorted(weights, reverse=True)[:3])
        
        analysis["max_position_weight"] = max_weight
        analysis["top3_concentration"] = top3_weight
        
        # Geographic diversification (simplified - would need sector/region data)
        analysis["geographic_diversification"] = "Unknown - requires additional data"
        
        # Sector diversification (simplified)
        analysis["sector_diversification"] = "Unknown - requires sector classification"
        
        return analysis
    
    def _calculate_diversification_score(self, diversification_analysis: Dict[str, Any]) -> int:
        """Calculate diversification score (0-100)"""
        
        score = 0
        
        # Position count score (40 points max)
        position_count = diversification_analysis.get("position_count", 0)
        if position_count >= 15:
            score += 40
        elif position_count >= 10:
            score += 30
        elif position_count >= 5:
            score += 20
        elif position_count >= 3:
            score += 10
        
        # Concentration score (60 points max)
        max_weight = diversification_analysis.get("max_position_weight", 0)
        top3_weight = diversification_analysis.get("top3_concentration", 0)
        
        if max_weight < 10:
            score += 30
        elif max_weight < 15:
            score += 25
        elif max_weight < 20:
            score += 15
        elif max_weight < 30:
            score += 5
        
        if top3_weight < 40:
            score += 30
        elif top3_weight < 50:
            score += 20
        elif top3_weight < 60:
            score += 10
        
        return min(100, score)
    
    def _generate_diversification_suggestions(self, diversification_analysis: Dict[str, Any]) -> List[str]:
        """Generate diversification improvement suggestions"""
        
        suggestions = []
        
        position_count = diversification_analysis.get("position_count", 0)
        max_weight = diversification_analysis.get("max_position_weight", 0)
        top3_weight = diversification_analysis.get("top3_concentration", 0)
        
        # Position count suggestions
        if position_count < 5:
            suggestions.append("Increase portfolio size to at least 5-10 positions")
        elif position_count < 10:
            suggestions.append("Consider adding 2-3 more positions for better diversification")
        
        # Concentration suggestions
        if max_weight > 25:
            suggestions.append(f"Reduce largest position (currently {max_weight:.1f}%) to under 20%")
        elif max_weight > 20:
            suggestions.append("Monitor largest position size for concentration risk")
        
        if top3_weight > 60:
            suggestions.append("Top 3 positions are too concentrated - consider rebalancing")
        
        return suggestions
    
    def _suggest_new_positions(self, portfolio: Dict[str, Any], 
                              diversification_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest new positions to improve diversification"""
        
        suggestions = []
        current_symbols = set(portfolio.keys())
        
        # Sector suggestions (simplified)
        if "AAPL" in current_symbols or "GOOGL" in current_symbols:
            if not any(symbol in ["JPM", "BAC", "WFC"] for symbol in current_symbols):
                suggestions.append({
                    "sector": "Financial",
                    "rationale": "Add financial sector exposure",
                    "examples": ["JPM", "BAC", "WFC"]
                })
        
        if not any(symbol in ["JNJ", "PFE", "UNH"] for symbol in current_symbols):
            suggestions.append({
                "sector": "Healthcare",
                "rationale": "Add defensive healthcare exposure",
                "examples": ["JNJ", "PFE", "UNH"]
            })
        
        # International exposure
        suggestions.append({
            "sector": "International",
            "rationale": "Consider international diversification",
            "examples": ["VEA", "VWO", "VXUS"]
        })
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _prioritize_diversification_actions(self, recommendations: List[str]) -> List[Dict[str, str]]:
        """Prioritize diversification actions"""
        
        priorities = []
        
        for i, rec in enumerate(recommendations):
            if "concentration" in rec.lower() or "largest position" in rec.lower():
                priority = "High"
            elif "position" in rec.lower() and "add" in rec.lower():
                priority = "Medium"
            else:
                priority = "Low"
            
            priorities.append({
                "action": rec,
                "priority": priority,
                "timeframe": "1-3 months" if priority == "High" else "3-6 months"
            })
        
        return sorted(priorities, key=lambda x: {"High": 3, "Medium": 2, "Low": 1}[x["priority"]], reverse=True)
    
    async def _calculate_investment_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate investment score for ranking"""
        
        key_metrics = analysis.get("key_metrics", {})
        technical_analysis = analysis.get("technical_analysis", {})
        company_info = analysis.get("company_info", {})
        
        return await self._calculate_recommendation_score(key_metrics, technical_analysis, company_info)
    
    def _rank_investment_options(self, stock_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Rank investment options by score"""
        
        ranked = []
        for symbol, score in stock_scores.items():
            ranked.append({
                "symbol": symbol,
                "score": score,
                "rank": 0  # Will be set after sorting
            })
        
        # Sort by score (highest first)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank
        for i, item in enumerate(ranked):
            item["rank"] = i + 1
        
        return ranked
    
    def _generate_comparison_insights(self, individual_analyses: Dict[str, Any], 
                                   ranked_stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from stock comparison"""
        
        insights = {}
        
        if ranked_stocks:
            best_stock = ranked_stocks[0]
            worst_stock = ranked_stocks[-1] if len(ranked_stocks) > 1 else None
            
            insights["top_performer"] = {
                "symbol": best_stock["symbol"],
                "score": best_stock["score"],
                "reason": "Highest overall investment score"
            }
            
            if worst_stock:
                insights["lowest_rated"] = {
                    "symbol": worst_stock["symbol"],
                    "score": worst_stock["score"],
                    "reason": "Lowest overall investment score"
                }
            
            # Score distribution
            scores = [stock["score"] for stock in ranked_stocks]
            insights["score_range"] = {
                "highest": max(scores),
                "lowest": min(scores),
                "average": sum(scores) / len(scores),
                "spread": max(scores) - min(scores)
            }
        
        return insights
    
    def _generate_allocation_recommendations(self, ranked_stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate portfolio allocation recommendations"""
        
        recommendations = []
        
        if not ranked_stocks:
            return recommendations
        
        # Allocate weights based on ranking (top stocks get higher allocation)
        total_stocks = len(ranked_stocks)
        
        for stock in ranked_stocks:
            rank = stock["rank"]
            score = stock["score"]
            
            # Weight allocation based on score and rank
            if rank == 1 and score > 20:
                allocation = 25  # Top pick gets 25%
            elif rank <= 3 and score > 10:
                allocation = 20  # Top 3 get 20%
            elif score > 0:
                allocation = 15  # Positive scores get 15%
            elif score > -10:
                allocation = 10  # Neutral scores get 10%
            else:
                allocation = 5   # Negative scores get 5%
            
            recommendation = "Strong Buy" if score > 20 else "Buy" if score > 10 else "Hold" if score > -10 else "Avoid"
            
            recommendations.append({
                "symbol": stock["symbol"],
                "rank": rank,
                "score": score,
                "suggested_allocation": allocation,
                "recommendation": recommendation
            })
        
        return recommendations
    
    def _analyze_sentiment_signals(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment for market timing signals"""
        
        market_sentiment = sentiment_data.get("market_sentiment", {})
        sentiment_score = market_sentiment.get("score", 0)
        sector_sentiment = sentiment_data.get("sector_sentiment", {})
        
        signals = {}
        
        # Overall market sentiment signal
        if sentiment_score > 0.3:
            signals["market_sentiment"] = "Bullish"
        elif sentiment_score < -0.3:
            signals["market_sentiment"] = "Bearish"
        else:
            signals["market_sentiment"] = "Neutral"
        
        # Sector rotation signals
        if sector_sentiment:
            best_sector = max(sector_sentiment.items(), key=lambda x: x[1])
            worst_sector = min(sector_sentiment.items(), key=lambda x: x[1])
            
            signals["sector_rotation"] = {
                "favored_sector": best_sector[0],
                "avoided_sector": worst_sector[0],
                "sentiment_spread": best_sector[1] - worst_sector[1]
            }
        
        return signals
    
    def _generate_timing_signals(self, sentiment_signals: Dict[str, Any], 
                               market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate market timing signals"""
        
        signals = {}
        
        market_sentiment = sentiment_signals.get("market_sentiment", "Neutral")
        
        # Basic timing signals based on sentiment
        if market_sentiment == "Bullish":
            signals["market_timing"] = "Favorable for buying"
            signals["cash_level"] = "Low (10-20%)"
            signals["equity_exposure"] = "High (80-90%)"
        elif market_sentiment == "Bearish":
            signals["market_timing"] = "Consider defensive positioning"
            signals["cash_level"] = "High (30-50%)"
            signals["equity_exposure"] = "Low (50-70%)"
        else:
            signals["market_timing"] = "Neutral - maintain balanced approach"
            signals["cash_level"] = "Moderate (20-30%)"
            signals["equity_exposure"] = "Moderate (70-80%)"
        
        # Sector timing
        sector_rotation = sentiment_signals.get("sector_rotation", {})
        if sector_rotation:
            signals["sector_timing"] = {
                "overweight": sector_rotation.get("favored_sector", "Technology"),
                "underweight": sector_rotation.get("avoided_sector", "Energy")
            }
        
        return signals
    
    def _calculate_signal_strength(self, timing_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate signal strength and confidence"""
        
        market_timing = timing_signals.get("market_timing", "")
        
        if "favorable" in market_timing.lower():
            strength = "Strong"
            confidence = 75
        elif "defensive" in market_timing.lower():
            strength = "Strong"
            confidence = 70
        elif "neutral" in market_timing.lower():
            strength = "Weak"
            confidence = 40
        else:
            strength = "Moderate"
            confidence = 60
        
        return {
            "signal_strength": strength,
            "confidence": confidence,
            "reliability": "Moderate" if confidence > 60 else "Low"
        }
    
    def _generate_market_outlook_recommendation(self, timing_signals: Dict[str, Any]) -> str:
        """Generate market outlook recommendation"""
        
        market_timing = timing_signals.get("market_timing", "")
        equity_exposure = timing_signals.get("equity_exposure", "")
        
        if "favorable" in market_timing.lower():
            return "Positive market outlook supports higher equity allocation and growth-oriented positioning"
        elif "defensive" in market_timing.lower():
            return "Cautious market outlook suggests defensive positioning and higher cash allocation"
        else:
            return "Mixed market signals suggest maintaining balanced portfolio allocation"
    
    def _generate_timing_action_recommendations(self, timing_signals: Dict[str, Any]) -> List[str]:
        """Generate specific timing-based action recommendations"""
        
        actions = []
        
        market_timing = timing_signals.get("market_timing", "")
        cash_level = timing_signals.get("cash_level", "")
        sector_timing = timing_signals.get("sector_timing", {})
        
        # Cash level actions
        if "high" in cash_level.lower():
            actions.append("Increase cash position to 30-50% for defensive positioning")
        elif "low" in cash_level.lower():
            actions.append("Reduce cash to 10-20% to maximize equity exposure")
        
        # Sector rotation actions
        if sector_timing:
            overweight = sector_timing.get("overweight", "")
            underweight = sector_timing.get("underweight", "")
            
            if overweight:
                actions.append(f"Consider overweighting {overweight} sector")
            if underweight:
                actions.append(f"Consider underweighting {underweight} sector")
        
        # General market actions
        if "favorable" in market_timing.lower():
            actions.append("Consider adding to growth positions")
        elif "defensive" in market_timing.lower():
            actions.append("Consider adding to defensive positions (utilities, consumer staples)")
        
        return actions[:5]  # Limit to top 5 actions

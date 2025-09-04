"""
Orchestrator Agent - Manages workflow and coordinates other agents
"""
import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings

# Import your actual agents - adjust imports as needed
from agents.data_analyst_agent import DataAnalystAgent
from agents.news_sentiment_agent import NewsSentimentAgent


class OrchestratorAgent:
    """Main orchestrator that manages the workflow between agents"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["orchestrator"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["orchestrator"]["system_message"]
        )
        
        # Initialize child agents
        self.data_analyst = DataAnalystAgent(model_client)
        self.news_sentiment = NewsSentimentAgent(model_client)
        
        # Agent registry for dynamic calls
        self.agents = {
            "data_analyst": self.data_analyst,
            "news_sentiment": self.news_sentiment
        }
        
    async def process_query(self, user_query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query and coordinate agents - THIS IS THE MISSING METHOD"""
        
        try:
            if user_context is None:
                user_context = {}
            
            # Analyze the query to determine which agents are needed
            query_analysis = await self._analyze_query(user_query)
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(query_analysis, user_context)
            
            # Execute the plan
            results = await self._execute_plan(execution_plan, user_query)
            
            # Generate final response
            final_response = await self._generate_final_response(user_query, query_analysis, results)
            
            return {
                "query": user_query,
                "query_analysis": query_analysis,
                "execution_plan": execution_plan,
                "agent_results": results,
                "final_response": final_response,
                "agents_involved": execution_plan.get("agents_needed", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Orchestrator failed to process query: {str(e)}",
                "query": user_query,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query to understand requirements"""
        
        try:
            # Use simple rule-based analysis (could be enhanced with LLM)
            analysis = {
                "symbols": self._extract_symbols(user_query),
                "analysis_type": self._determine_analysis_type(user_query),
                "time_period": self._extract_time_period(user_query),
                "data_needed": self._determine_data_needed(user_query),
                "output_format": self._determine_output_format(user_query),
                "intent": self._classify_intent(user_query),
                "complexity": self._assess_complexity(user_query)
            }
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Query analysis failed: {str(e)}",
                "symbols": [],
                "analysis_type": "general_analysis",
                "time_period": "default",
                "data_needed": ["stock_price"],
                "output_format": "text",
                "intent": "unknown",
                "complexity": "low"
            }
    
    async def _create_execution_plan(self, analysis: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan based on query analysis"""
        
        try:
            plan = {
                "agents_needed": [],
                "execution_order": [],
                "data_dependencies": {},
                "expected_outputs": [],
                "parallel_execution": False
            }
            
            # Determine which agents are needed based on analysis
            analysis_type = analysis.get("analysis_type", "")
            symbols = analysis.get("symbols", [])
            data_needed = analysis.get("data_needed", [])
            
            # Always need data analyst for stock-related queries
            if symbols or analysis_type in ["price_analysis", "comparison", "portfolio"]:
                plan["agents_needed"].append("data_analyst")
                plan["expected_outputs"].append("stock_analysis")
            
            # Need sentiment analysis for news-related queries
            if ("news" in analysis_type or "sentiment" in analysis_type or 
                "news" in data_needed or analysis.get("intent") == "sentiment"):
                plan["agents_needed"].append("news_sentiment")
                plan["expected_outputs"].append("sentiment_analysis")
            
            # Define execution order based on dependencies
            if "data_analyst" in plan["agents_needed"] and "news_sentiment" in plan["agents_needed"]:
                # Can run in parallel since they're independent
                plan["execution_order"] = ["data_analyst", "news_sentiment"]
                plan["parallel_execution"] = True
            elif "data_analyst" in plan["agents_needed"]:
                plan["execution_order"] = ["data_analyst"]
            elif "news_sentiment" in plan["agents_needed"]:
                plan["execution_order"] = ["news_sentiment"]
            else:
                # Fallback to data analyst for general queries
                plan["agents_needed"] = ["data_analyst"]
                plan["execution_order"] = ["data_analyst"]
            
            return plan
            
        except Exception as e:
            # Fallback plan
            return {
                "agents_needed": ["data_analyst"],
                "execution_order": ["data_analyst"],
                "data_dependencies": {},
                "expected_outputs": ["stock_analysis"],
                "parallel_execution": False,
                "error": f"Plan creation failed: {str(e)}"
            }
    
    async def _execute_plan(self, plan: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Execute the planned workflow"""
        
        results = {}
        shared_context = {
            "query": user_query,
            "symbols": self._extract_symbols(user_query),
            "analysis_type": self._determine_analysis_type(user_query)
        }
        
        try:
            if plan.get("parallel_execution", False):
                # Execute agents in parallel
                tasks = []
                for agent_name in plan["agents_needed"]:
                    task = self._call_agent(agent_name, shared_context)
                    tasks.append((agent_name, task))
                
                # Wait for all tasks to complete
                for agent_name, task in tasks:
                    try:
                        agent_result = await task
                        results[agent_name] = agent_result
                        shared_context[f"{agent_name}_result"] = agent_result
                    except Exception as e:
                        results[agent_name] = {"error": f"Agent {agent_name} failed: {str(e)}"}
            else:
                # Execute agents sequentially
                for agent_name in plan["execution_order"]:
                    try:
                        agent_result = await self._call_agent(agent_name, shared_context)
                        results[agent_name] = agent_result
                        shared_context[f"{agent_name}_result"] = agent_result
                    except Exception as e:
                        results[agent_name] = {"error": f"Agent {agent_name} failed: {str(e)}"}
            
            return results
            
        except Exception as e:
            return {"execution_error": f"Plan execution failed: {str(e)}"}
    
    async def _call_agent(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific agent based on context"""
        
        try:
            query = context.get("query", "")
            symbols = context.get("symbols", [])
            analysis_type = context.get("analysis_type", "")
            
            if agent_name == "data_analyst":
                return await self._call_data_analyst(query, symbols, analysis_type)
            elif agent_name == "news_sentiment":
                return await self._call_news_sentiment(query, symbols, analysis_type)
            else:
                return {"error": f"Unknown agent: {agent_name}"}
                
        except Exception as e:
            return {"error": f"Agent call failed: {str(e)}"}
    
    async def _call_data_analyst(self, query: str, symbols: List[str], analysis_type: str) -> Dict[str, Any]:
        """Call data analyst agent with appropriate method"""
        
        try:
            if analysis_type == "comparison" and len(symbols) > 1:
                return await self.data_analyst.compare_stocks(symbols[:5])
            elif symbols:
                return await self.data_analyst.analyze_stock(symbols[0])
            elif "portfolio" in analysis_type:
                # Mock portfolio for demonstration
                mock_portfolio = {"AAPL": {"shares": 100, "avg_cost": 150.0}}
                return await self.data_analyst.get_portfolio_analysis(mock_portfolio)
            elif "earnings" in query.lower():
                symbol = symbols[0] if symbols else "AAPL"
                return await self.data_analyst.get_earnings_analysis(symbol)
            else:
                # Default to single stock analysis
                symbol = symbols[0] if symbols else "AAPL"
                return await self.data_analyst.analyze_stock(symbol)
                
        except Exception as e:
            return {"error": f"Data analyst call failed: {str(e)}"}
    
    async def _call_news_sentiment(self, query: str, symbols: List[str], analysis_type: str) -> Dict[str, Any]:
        """Call news sentiment agent with appropriate method"""
        
        try:
            if symbols and len(symbols) == 1:
                return await self.news_sentiment.analyze_stock_sentiment(symbols[0])
            elif symbols and len(symbols) > 1:
                return await self.news_sentiment.compare_stock_sentiments(symbols)
            else:
                return await self.news_sentiment.analyze_market_sentiment()
                
        except Exception as e:
            return {"error": f"News sentiment call failed: {str(e)}"}
    
    async def _generate_final_response(self, query: str, analysis: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final coordinated response"""
        
        try:
            final_response = {
                "summary": f"Analysis complete for query: '{query}'",
                "key_findings": [],
                "recommendations": [],
                "data_sources": [],
                "confidence": 0.0
            }
            
            # Extract key findings from each agent result
            if "data_analyst" in results and "error" not in results["data_analyst"]:
                data_result = results["data_analyst"]
                if "recommendation" in data_result:
                    rec = data_result["recommendation"]
                    final_response["recommendations"].append({
                        "type": "investment",
                        "action": rec.get("action", "HOLD"),
                        "confidence": rec.get("confidence", 0.5),
                        "reasoning": rec.get("reasoning", [])
                    })
                
                final_response["data_sources"].append("market_data")
                
            if "news_sentiment" in results and "error" not in results["news_sentiment"]:
                sentiment_result = results["news_sentiment"]
                if "overall_sentiment" in sentiment_result:
                    sentiment = sentiment_result["overall_sentiment"]
                    final_response["key_findings"].append({
                        "type": "sentiment",
                        "value": sentiment.get("label", "neutral"),
                        "score": sentiment.get("score", 0),
                        "confidence": sentiment.get("confidence", 0)
                    })
                
                final_response["data_sources"].append("news_sentiment")
            
            # Calculate overall confidence
            confidences = []
            for rec in final_response["recommendations"]:
                confidences.append(rec.get("confidence", 0))
            for finding in final_response["key_findings"]:
                confidences.append(finding.get("confidence", 0))
            
            if confidences:
                final_response["confidence"] = sum(confidences) / len(confidences)
            
            return final_response
            
        except Exception as e:
            return {
                "summary": f"Analysis completed with errors for query: '{query}'",
                "error": f"Response generation failed: {str(e)}",
                "partial_results": True
            }
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        
        # Company name to symbol mapping
        symbol_mapping = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
            'amazon': 'AMZN', 'tesla': 'TSLA', 'netflix': 'NFLX', 'facebook': 'META',
            'meta': 'META', 'nvidia': 'NVDA', 'intel': 'INTC', 'amd': 'AMD'
        }
        
        query_lower = query.lower()
        symbols = []
        
        # Look for direct symbols (2-5 uppercase letters)
        direct_symbols = re.findall(r'\b[A-Z]{2,5}\b', query)
        common_words = {"THE", "AND", "OR", "IS", "ARE", "VS", "THIS", "THAT", "WHAT", "HOW", "WHEN"}
        symbols.extend([s for s in direct_symbols if s not in common_words])
        
        # Look for company names
        for name, symbol in symbol_mapping.items():
            if name in query_lower and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols[:5]  # Limit to 5 symbols
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis requested"""
        
        query_lower = query.lower()
        
        if "compare" in query_lower or " vs " in query_lower or "versus" in query_lower:
            return "comparison"
        elif "portfolio" in query_lower or "diversif" in query_lower:
            return "portfolio"
        elif "news" in query_lower or "sentiment" in query_lower:
            return "news_sentiment"
        elif "earnings" in query_lower or "quarterly" in query_lower:
            return "earnings"
        elif "price" in query_lower or "stock" in query_lower:
            return "price_analysis"
        elif "risk" in query_lower:
            return "risk_analysis"
        elif "recommend" in query_lower or "suggest" in query_lower or "advice" in query_lower:
            return "recommendation"
        else:
            return "general_analysis"
    
    def _extract_time_period(self, query: str) -> str:
        """Extract time period from query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["quarter", "q1", "q2", "q3", "q4", "quarterly"]):
            return "quarterly"
        elif any(word in query_lower for word in ["year", "annual", "yearly", "ytd"]):
            return "yearly"
        elif any(word in query_lower for word in ["month", "monthly"]):
            return "monthly"
        elif any(word in query_lower for word in ["week", "weekly"]):
            return "weekly"
        elif any(word in query_lower for word in ["day", "daily", "today"]):
            return "daily"
        else:
            return "default"
    
    def _determine_data_needed(self, query: str) -> List[str]:
        """Determine what data is needed"""
        
        query_lower = query.lower()
        data_needed = []
        
        if "earnings" in query_lower:
            data_needed.append("earnings")
        if "price" in query_lower:
            data_needed.append("stock_price")
        if "news" in query_lower:
            data_needed.append("news")
        if "fundamental" in query_lower or "financial" in query_lower:
            data_needed.append("fundamentals")
        if "chart" in query_lower or "graph" in query_lower:
            data_needed.append("historical_data")
        if "sentiment" in query_lower:
            data_needed.append("sentiment")
        
        return data_needed if data_needed else ["stock_price"]
    
    def _determine_output_format(self, query: str) -> str:
        """Determine preferred output format"""
        
        query_lower = query.lower()
        
        if "chart" in query_lower or "graph" in query_lower or "plot" in query_lower:
            return "chart"
        elif "report" in query_lower or "pdf" in query_lower:
            return "report"
        elif "summary" in query_lower or "brief" in query_lower:
            return "summary"
        elif "table" in query_lower:
            return "table"
        else:
            return "text"
    
    def _classify_intent(self, query: str) -> str:
        """Classify the user's intent"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["buy", "sell", "invest", "purchase"]):
            return "investment_decision"
        elif any(word in query_lower for word in ["compare", "versus", "better"]):
            return "comparison"
        elif any(word in query_lower for word in ["news", "sentiment", "feeling"]):
            return "sentiment"
        elif any(word in query_lower for word in ["portfolio", "allocation", "diversify"]):
            return "portfolio_management"
        elif any(word in query_lower for word in ["analyze", "analysis", "how is"]):
            return "analysis"
        else:
            return "information_seeking"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        
        symbols = self._extract_symbols(query)
        data_types = len(self._determine_data_needed(query))
        
        if len(symbols) > 2 or data_types > 3:
            return "high"
        elif len(symbols) > 1 or data_types > 1:
            return "medium"
        else:
            return "low"

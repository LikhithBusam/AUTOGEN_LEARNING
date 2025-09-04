"""
Orchestrator Agent - Manages workflow and coordinates other agents
"""
import asyncio
from typing import Dict, List, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from config.settings import settings

class OrchestratorAgent:
    """Main orchestrator that manages the workflow between agents"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.model_client = model_client
        self.agent = AssistantAgent(
            name=settings.agent_config["orchestrator"]["name"],
            model_client=model_client,
            system_message=settings.agent_config["orchestrator"]["system_message"]
        )
        
    async def process_query(self, user_query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query and coordinate agents"""
        
        # Analyze the query to determine which agents are needed
        query_analysis = await self._analyze_query(user_query)
        
        # Create execution plan
        execution_plan = await self._create_execution_plan(query_analysis, user_context)
        
        # Execute the plan
        results = await self._execute_plan(execution_plan)
        
        return {
            "query": user_query,
            "analysis": query_analysis,
            "execution_plan": execution_plan,
            "results": results,
            "agents_involved": execution_plan.get("agents_needed", [])
        }
    
    async def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query to understand requirements"""
        
        analysis_prompt = f"""
        Analyze this financial query and determine what information and actions are needed:
        
        Query: "{user_query}"
        
        Please identify:
        1. Stock symbols mentioned (if any)
        2. Type of analysis requested (price, comparison, news, portfolio, etc.)
        3. Time period (if specified)
        4. Specific data needed (earnings, news, charts, etc.)
        5. Output format preferred (report, chart, summary, etc.)
        
        Respond in JSON format with these categories.
        """
        
        message = TextMessage(content=analysis_prompt, source="user")
        result = await self.agent.on_messages([message])
        
        # Parse the response (simplified - in production would use proper JSON parsing)
        analysis = {
            "symbols": self._extract_symbols(user_query),
            "analysis_type": self._determine_analysis_type(user_query),
            "time_period": self._extract_time_period(user_query),
            "data_needed": self._determine_data_needed(user_query),
            "output_format": self._determine_output_format(user_query)
        }
        
        return analysis
    
    async def _create_execution_plan(self, analysis: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan based on query analysis"""
        
        plan = {
            "agents_needed": [],
            "execution_order": [],
            "data_flow": {},
            "expected_outputs": []
        }
        
        # Determine which agents are needed based on analysis
        if analysis.get("symbols") or analysis.get("analysis_type") in ["price", "comparison", "portfolio"]:
            plan["agents_needed"].append("data_analyst")
        
        if "news" in analysis.get("analysis_type", "") or "sentiment" in user_context.get("preferences", {}):
            plan["agents_needed"].append("news_sentiment")
        
        if analysis.get("output_format") in ["chart", "graph", "visualization"]:
            plan["agents_needed"].append("visualization")
        
        if analysis.get("analysis_type") in ["recommendation", "portfolio", "risk"]:
            plan["agents_needed"].append("recommendation")
        
        if analysis.get("output_format") in ["report", "pdf", "summary"]:
            plan["agents_needed"].append("report_generator")
        
        # Define execution order (data collection first, then analysis, then output)
        order_priority = {
            "data_analyst": 1,
            "news_sentiment": 2,
            "recommendation": 3,
            "visualization": 4,
            "report_generator": 5
        }
        
        plan["execution_order"] = sorted(
            plan["agents_needed"], 
            key=lambda x: order_priority.get(x, 999)
        )
        
        return plan
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned workflow"""
        
        results = {}
        shared_context = {}
        
        for agent_name in plan["execution_order"]:
            try:
                # Each agent would be implemented to process based on shared context
                agent_result = await self._call_agent(agent_name, shared_context)
                results[agent_name] = agent_result
                
                # Update shared context with results
                shared_context[agent_name] = agent_result
                
            except Exception as e:
                results[agent_name] = {"error": str(e)}
        
        return results
    
    async def _call_agent(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific agent (placeholder - would import and call actual agents)"""
        
        # This would import and call the actual agent implementations
        # For now, returning a placeholder
        return {
            "agent": agent_name,
            "status": "success",
            "data": f"Processed by {agent_name}",
            "context_used": list(context.keys())
        }
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        # Simple implementation - would use NLP in production
        import re
        
        # Look for potential stock symbols (uppercase 1-5 letters)
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query)
        
        # Filter common words that aren't symbols
        common_words = {"THE", "AND", "OR", "IS", "ARE", "VS", "THIS", "THAT", "WHAT", "HOW", "WHEN"}
        symbols = [s for s in symbols if s not in common_words]
        
        return symbols
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis requested"""
        query_lower = query.lower()
        
        if "compare" in query_lower or "vs" in query_lower:
            return "comparison"
        elif "portfolio" in query_lower:
            return "portfolio"
        elif "news" in query_lower or "sentiment" in query_lower:
            return "news_sentiment"
        elif "price" in query_lower or "stock" in query_lower:
            return "price_analysis"
        elif "risk" in query_lower:
            return "risk_analysis"
        elif "recommend" in query_lower or "suggest" in query_lower:
            return "recommendation"
        else:
            return "general_analysis"
    
    def _extract_time_period(self, query: str) -> str:
        """Extract time period from query"""
        query_lower = query.lower()
        
        if "quarter" in query_lower or "q1" in query_lower or "q2" in query_lower:
            return "quarterly"
        elif "year" in query_lower or "annual" in query_lower:
            return "yearly"
        elif "month" in query_lower:
            return "monthly"
        elif "week" in query_lower:
            return "weekly"
        elif "day" in query_lower or "today" in query_lower:
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
        
        return data_needed if data_needed else ["stock_price"]
    
    def _determine_output_format(self, query: str) -> str:
        """Determine preferred output format"""
        query_lower = query.lower()
        
        if "chart" in query_lower or "graph" in query_lower:
            return "chart"
        elif "report" in query_lower or "pdf" in query_lower:
            return "report"
        elif "summary" in query_lower:
            return "summary"
        else:
            return "text"

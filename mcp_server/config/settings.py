"""
Configuration settings for the MCP-Powered Financial Analyst
"""
import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    yahoo_finance_api_key: str = os.getenv("YAHOO_FINANCE_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/financial_analyst.db")
    
    # MCP Server
    mcp_server_host: str = os.getenv("MCP_SERVER_HOST", "localhost")
    mcp_server_port: int = int(os.getenv("MCP_SERVER_PORT", "8000"))
    
    # Output Directories
    reports_output_dir: str = os.getenv("REPORTS_OUTPUT_DIR", "./reports")
    charts_output_dir: str = os.getenv("CHARTS_OUTPUT_DIR", "./reports/charts")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./data/financial_analyst.log")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "financial-analyst-secret-key")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Agent Configuration
    agent_config: Dict[str, Any] = {
        "orchestrator": {
            "name": "OrchestratorAgent",
            "system_message": """You are the Orchestrator Agent for a financial analysis system. 
            Your role is to coordinate between different specialized agents to fulfill user requests.
            You manage the workflow and ensure all agents work together efficiently.""",
            "temperature": 0.3
        },
        "data_analyst": {
            "name": "DataAnalystAgent", 
            "system_message": """You are a Data Analyst Agent specializing in financial data retrieval and analysis.
            You can access real-time financial data through MCP integrations with various APIs.
            Provide accurate, timely financial data and perform quantitative analysis.""",
            "temperature": 0.2
        },
        "news_sentiment": {
            "name": "NewsSentimentAgent",
            "system_message": """You are a News & Sentiment Analysis Agent.
            You gather financial news and perform sentiment analysis to gauge market sentiment.
            Provide insights on how news might impact stock prices and market trends.""",
            "temperature": 0.4
        },
        "report_generator": {
            "name": "ReportGeneratorAgent",
            "system_message": """You are a Report Generator Agent that creates structured financial reports.
            You compile data from other agents into comprehensive PDF and HTML reports.
            Focus on clear, professional formatting and actionable insights.""",
            "temperature": 0.3
        },
        "visualization": {
            "name": "VisualizationAgent",
            "system_message": """You are a Visualization Agent that creates charts and graphs.
            You generate various types of financial visualizations including candlestick charts,
            trend analysis, portfolio allocations, and risk assessment charts.""",
            "temperature": 0.2
        },
        "recommendation": {
            "name": "RecommendationAgent",
            "system_message": """You are a Recommendation Agent that provides investment advice.
            You analyze risk, perform portfolio optimization, and suggest investment strategies
            based on Modern Portfolio Theory and risk-adjusted returns.""",
            "temperature": 0.3
        }
    }
    
    # Financial Data Sources
    financial_apis: List[str] = [
        "alpha_vantage",
        "yahoo_finance", 
        "financial_modeling_prep"
    ]
    
    # News Sources
    news_sources: List[str] = [
        "newsapi",
        "financial_times",
        "reuters",
        "bloomberg"
    ]
    
    # Default Analysis Parameters
    default_analysis_period: str = "1y"  # 1 year
    default_risk_free_rate: float = 0.02  # 2%
    monte_carlo_simulations: int = 10000
    portfolio_rebalance_frequency: str = "quarterly"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validation
def validate_settings():
    """Validate that required settings are present"""
    required_keys = [
        "google_api_key",
        "alpha_vantage_api_key", 
        "news_api_key"
    ]
    
    missing_keys = []
    for key in required_keys:
        if not getattr(settings, key):
            missing_keys.append(key.upper())
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    return True

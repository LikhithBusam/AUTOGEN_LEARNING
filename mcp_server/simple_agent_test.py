import sys

agents = [
    "orchestrator_agent",
    "data_analyst_agent", 
    "news_sentiment_agent",
    "visualization_agent",
    "recommendation_agent",
    "report_generator_agent"
]

for agent in agents:
    try:
        exec(f"import agents.{agent}")
        print(f"✅ {agent} - OK")
    except Exception as e:
        print(f"❌ {agent} - FAILED: {e}")

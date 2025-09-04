#!/usr/bin/env python3
"""Test all agents import correctly"""

def test_agent_imports():
    agents = [
        "orchestrator_agent",
        "data_analyst_agent", 
        "news_sentiment_agent",
        "visualization_agent",
        "recommendation_agent", 
        "report_generator_agent"
    ]
    
    for agent_name in agents:
        try:
            module_name = f"agents.{agent_name}"
            __import__(module_name)
            print(f"✓ {agent_name}: OK")
        except Exception as e:
            print(f"✗ {agent_name}: FAILED - {str(e)}")
    
    print("\n=== Agent Import Test Complete ===")

if __name__ == "__main__":
    test_agent_imports()

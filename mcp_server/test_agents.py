#!/usr/bin/env python3
"""
Individual Agent Import Test
Tests each agent separately to identify which ones are working
"""
import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_agent_import(agent_module, agent_class):
    """Test importing a specific agent"""
    try:
        module = __import__(agent_module, fromlist=[agent_class])
        agent_cls = getattr(module, agent_class)
        print(f"✅ {agent_module}.{agent_class} - Import OK")
        return True, agent_cls
    except Exception as e:
        print(f"❌ {agent_module}.{agent_class} - Import FAILED")
        print(f"   Error: {str(e)}")
        return False, None

def test_agent_creation(agent_cls, agent_name):
    """Test creating an agent instance"""
    try:
        from utils.model_client import create_gemini_model_client
        model_client = create_gemini_model_client()
        agent = agent_cls(model_client)
        print(f"✅ {agent_name} - Creation OK")
        return True, agent
    except Exception as e:
        print(f"❌ {agent_name} - Creation FAILED")
        print(f"   Error: {str(e)}")
        return False, None

def main():
    print("=== Individual Agent Test ===")
    
    # List of agents to test
    agents_to_test = [
        ("agents.orchestrator_agent", "OrchestratorAgent"),
        ("agents.data_analyst_agent", "DataAnalystAgent"), 
        ("agents.news_sentiment_agent", "NewsSentimentAgent"),
        ("agents.visualization_agent", "VisualizationAgent"),
        ("agents.recommendation_agent", "RecommendationAgent"),
        ("agents.report_generator_agent", "ReportGeneratorAgent"),
    ]
    
    import_results = {}
    creation_results = {}
    
    print("\n--- Testing Imports ---")
    for module_name, class_name in agents_to_test:
        success, agent_cls = test_agent_import(module_name, class_name)
        import_results[class_name] = (success, agent_cls)
    
    print("\n--- Testing Agent Creation ---")
    for agent_name, (import_success, agent_cls) in import_results.items():
        if import_success:
            creation_success, agent_instance = test_agent_creation(agent_cls, agent_name)
            creation_results[agent_name] = creation_success
        else:
            creation_results[agent_name] = False
            print(f"⏭️  {agent_name} - Skipped (import failed)")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Import Results:")
    for agent_name, (success, _) in import_results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"  {agent_name}: {status}")
    
    print(f"\nCreation Results:")
    for agent_name, success in creation_results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"  {agent_name}: {status}")
    
    successful_imports = sum(1 for success, _ in import_results.values() if success)
    successful_creations = sum(1 for success in creation_results.values() if success)
    
    print(f"\nOverall Status:")
    print(f"✅ Successful imports: {successful_imports}/{len(agents_to_test)}")
    print(f"✅ Successful creations: {successful_creations}/{len(agents_to_test)}")

if __name__ == "__main__":
    main()

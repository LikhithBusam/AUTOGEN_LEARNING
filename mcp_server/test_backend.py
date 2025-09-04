#!/usr/bin/env python3
"""
Backend Module Import Test
Tests all key backend modules for import errors and basic functionality
"""
import sys
import os
import traceback
import importlib

# Add the mcp_server directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module_import(module_name):
    """Test importing a module and return result"""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} - Import OK")
        return True, module
    except Exception as e:
        print(f"❌ {module_name} - Import FAILED")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False, None

def test_basic_functionality():
    """Test basic functionality of key modules"""
    print("\n=== Testing Basic Functionality ===")
    
    # Test database module
    try:
        from data.database import db_manager, create_tables
        print("✅ Database module - Basic import OK")
        
        # Test if db_manager has required methods
        required_methods = ['init_db', 'save_user_query', 'get_cached_stock_data', 'cache_stock_data']
        for method in required_methods:
            if hasattr(db_manager, method):
                print(f"✅ db_manager.{method} - Method exists")
            else:
                print(f"❌ db_manager.{method} - Method missing")
    except Exception as e:
        print(f"❌ Database functionality test failed: {e}")
    
    # Test MCP client
    try:
        from mcp.financial_data_server import mcp_client, MCPClient
        print("✅ MCP client module - Basic import OK")
        
        # Test if mcp_client has required methods
        if hasattr(mcp_client, 'call_method'):
            print("✅ mcp_client.call_method - Method exists")
        else:
            print("❌ mcp_client.call_method - Method missing")
    except Exception as e:
        print(f"❌ MCP client functionality test failed: {e}")
    
    # Test model client
    try:
        from utils.model_client import create_gemini_model_client
        print("✅ Model client module - Basic import OK")
    except Exception as e:
        print(f"❌ Model client functionality test failed: {e}")

def main():
    print("=== Backend Module Import Test ===")
    
    # List of modules to test
    modules_to_test = [
        'config.settings',
        'data.database', 
        'utils.model_client',
        'mcp.financial_data_server',
        'agents.orchestrator_agent',
        'agents.data_analyst_agent',
        'agents.news_sentiment_agent',
        'agents.visualization_agent',
        'agents.report_generator_agent',
        'agents.recommendation_agent',
        'main'
    ]
    
    # Test imports
    success_count = 0
    for module_name in modules_to_test:
        success, _ = test_module_import(module_name)
        if success:
            success_count += 1
    
    print(f"\n=== Import Summary ===")
    print(f"✅ Successful imports: {success_count}/{len(modules_to_test)}")
    print(f"❌ Failed imports: {len(modules_to_test) - success_count}/{len(modules_to_test)}")
    
    # Test basic functionality
    test_basic_functionality()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Basic system test without requiring API keys
"""
import os
import sys

def test_basic_imports():
    """Test basic imports"""
    print("🔧 Testing basic imports...")
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import sqlalchemy
        print("✅ SQLAlchemy imported successfully")
    except ImportError as e:
        print(f"❌ SQLAlchemy import failed: {e}")
        return False
    
    return True

def test_module_structure():
    """Test if our modules can be imported"""
    print("\n📁 Testing module structure...")
    
    try:
        from config.settings import settings
        print("✅ Settings module imported")
        print(f"   Database URL: {settings.database_url}")
        print(f"   Log level: {settings.log_level}")
    except ImportError as e:
        print(f"❌ Settings import failed: {e}")
        return False
    
    try:
        from data.database import db_manager
        print("✅ Database module imported")
    except ImportError as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    try:
        from mcp.financial_data_server import mcp_client
        print("✅ MCP client imported")
    except ImportError as e:
        print(f"❌ MCP client import failed: {e}")
        return False
    
    return True

def test_database_init():
    """Test database initialization"""
    print("\n💾 Testing database initialization...")
    
    try:
        import asyncio
        from data.database import create_tables
        
        # Test async database init
        async def test_db():
            try:
                await create_tables()
                print("✅ Database tables created successfully")
                return True
            except Exception as e:
                print(f"❌ Database init failed: {e}")
                return False
        
        result = asyncio.run(test_db())
        return result
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_financial_data():
    """Test financial data retrieval (basic functionality)"""
    print("\n📊 Testing financial data capabilities...")
    
    try:
        import yfinance as yf
        
        # Test basic yfinance functionality
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info:
            print("✅ Financial data retrieval working")
            print(f"   Sample data: {info.get('longName', 'N/A')}")
            return True
        else:
            print("⚠️ No data retrieved but no error")
            return False
            
    except Exception as e:
        print(f"❌ Financial data test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Basic System Test\n")
    
    tests = [
        test_basic_imports,
        test_module_structure,
        test_database_init,
        test_financial_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the output above.")
        sys.exit(1)

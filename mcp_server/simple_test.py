"""
Simple test to check if basic imports work
"""
import sys
import os

print("🔍 Testing basic imports...")

try:
    # Test configuration
    print("Testing config...")
    from config.settings import settings
    print("✅ Settings imported")
    
    # Test FastAPI
    print("Testing FastAPI...")
    import fastapi
    import uvicorn
    print("✅ FastAPI and Uvicorn available")
    
    # Test AutoGen
    print("Testing AutoGen...")
    import autogen_agentchat
    import autogen_ext
    print("✅ AutoGen available")
    
    print("\n🎉 Basic imports successful!")
    print("Ready to run the main application!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

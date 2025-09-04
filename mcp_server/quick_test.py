#!/usr/bin/env python3
"""
Quick test to verify the FastAPI app starts correctly
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_creation():
    """Test that the FastAPI app can be created without errors"""
    try:
        print("Testing FastAPI app creation...")
        from main import app
        print("✅ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_startup_without_agents():
    """Test basic imports without starting agents"""
    try:
        print("Testing core imports...")
        from config.settings import settings
        from data.database import db_manager
        from utils.model_client import create_gemini_model_client
        print("✅ Core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Quick Backend Test ===")
    
    success_count = 0
    total_tests = 2
    
    if test_startup_without_agents():
        success_count += 1
    
    if test_app_creation():
        success_count += 1
    
    print(f"\n=== Results ===")
    print(f"✅ Tests passed: {success_count}/{total_tests}")
    print(f"❌ Tests failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 Backend is ready to run!")
        print("You can start the server with: python -m uvicorn main:app --reload")
    else:
        print("\n⚠️  Some issues found. Check the errors above.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test the AI-powered financial assistant to verify Gemini integration
"""

import sys
import os
sys.path.insert(0, '.')

def test_ai_integration():
    """Test if the AI components are properly initialized"""
    
    print("🔍 TESTING AI INTEGRATION")
    print("=" * 50)
    
    try:
        # Test model client import
        print("📦 Testing imports...")
        from utils.model_client import create_gemini_model_client
        from config.settings import settings
        print("✅ Model client import successful")
        
        # Test API key availability
        api_key = getattr(settings, 'google_api_key', None)
        if api_key:
            print(f"✅ Google API key found: {api_key[:10]}...{api_key[-5:]}")
        else:
            print("❌ Google API key not found")
            return
        
        # Test model client creation
        print("🤖 Testing Gemini client creation...")
        model_client = create_gemini_model_client()
        print("✅ Gemini model client created successfully")
        
        # Test data analyst initialization
        print("📊 Testing AI-powered data analyst...")
        from app import data_analyst
        print(f"✅ Data analyst AI mode: {data_analyst.use_ai}")
        
        if data_analyst.use_ai:
            print("🧠 AI mode is ENABLED - using Gemini for analysis")
        else:
            print("📝 AI mode is DISABLED - using rule-based fallback")
        
        # Test intelligent assistant
        print("🎯 Testing intelligent assistant...")
        from app import intelligent_assistant
        
        if intelligent_assistant:
            print("✅ Intelligent assistant initialized")
            
            # Test a simple query
            print("🔬 Testing AI analysis with simple query...")
            test_result = intelligent_assistant.process_query("Analyze Apple stock")
            
            analysis_method = test_result.get('analysis', {}).get('analysis_method', 'unknown')
            print(f"📈 Analysis method used: {analysis_method}")
            
            if test_result.get('sentence_format'):
                print("✅ Sentence format generated")
            if test_result.get('table_format'):
                print("✅ Table format generated")
                
            if 'error' in test_result:
                print(f"⚠️  Query processing had issues: {test_result['error']}")
            else:
                print("✅ Query processing successful")
                
        else:
            print("❌ Intelligent assistant not initialized")
        
        print("\n🎉 AI Integration Test Complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Suggestion: Check if all dependencies are installed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Suggestion: Check logs for detailed error information")

if __name__ == "__main__":
    test_ai_integration()

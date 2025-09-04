#!/usr/bin/env python3
"""
Quick test for news functionality
"""
import sys
import os
sys.path.insert(0, '.')

def test_quick():
    print("🚀 QUICK NEWS TEST")
    print("=" * 40)
    
    try:
        from app import intelligent_assistant
        
        # Test a simple news query
        result = intelligent_assistant.process_query("Get news for Apple")
        
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📈 Query Type: {result.get('query_type', 'Unknown')}")
        
        # Check sentence format
        sentence = result.get('sentence_format', '')
        print(f"💬 Response: {sentence[:100]}...")
        
        if result.get('success'):
            print("🎉 NEWS FUNCTIONALITY IS WORKING!")
        else:
            print("❌ Issues detected")
            if 'error' in result:
                print(f"Error: {result['error']}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_quick()

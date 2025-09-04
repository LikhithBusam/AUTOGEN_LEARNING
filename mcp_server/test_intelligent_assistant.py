#!/usr/bin/env python3
"""
Test the enhanced intelligent financial assistant with dual format outputs
"""

import sys
import os
sys.path.insert(0, '.')

from app import intelligent_assistant

def test_intelligent_assistant():
    """Test the intelligent assistant with various queries"""
    
    test_queries = [
        "Compare Apple vs Microsoft stocks",
        "Analyze Tesla stock",
        "Market overview today",
        "Should I buy NVIDIA?"
    ]
    
    print("🤖 TESTING INTELLIGENT FINANCIAL ASSISTANT")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📊 Test {i}: {query}")
        print("-" * 40)
        
        try:
            result = intelligent_assistant.process_query(query)
            
            print(f"✅ Success: {result.get('success', False)}")
            print(f"📈 Query Type: {result.get('query_type', 'Unknown')}")
            print(f"🎯 Symbols: {result.get('detected_symbols', [])}")
            
            # Display sentence format
            sentence = result.get('sentence_format', 'Not available')
            if sentence and sentence != 'Not available':
                print(f"\n💬 SENTENCE FORMAT:")
                print(sentence)
            
            # Display table format
            table = result.get('table_format', 'Not available')
            if table and table != 'Not available':
                print(f"\n📋 TABLE FORMAT:")
                print(table)
            
            # Display any errors
            if 'error' in result:
                print(f"\n❌ Error: {result['error']}")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    test_intelligent_assistant()

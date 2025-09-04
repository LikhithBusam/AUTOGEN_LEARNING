#!/usr/bin/env python3
"""
Test the enhanced stock report generation with real data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from app import generate_stock_report, normalize_stock_symbol

def test_stock_reports():
    """Test stock report generation with various symbols"""
    
    print("🧪 TESTING ENHANCED STOCK REPORTS")
    print("=" * 60)
    
    # Test various symbol formats
    test_symbols = [
        "tesla",      # Should convert to TSLA
        "TSLA",       # Already correct
        "apple",      # Should convert to AAPL  
        "AAPL",       # Already correct
        "google",     # Should convert to GOOGL
        "microsoft",  # Should convert to MSFT
        "INVALIDXYZ"  # Invalid symbol to test error handling
    ]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing symbol: '{symbol}'")
        print("-" * 40)
        
        try:
            # Test symbol normalization first
            normalized = normalize_stock_symbol(symbol)
            print(f"Symbol normalized: {symbol} -> {normalized}")
            
            # Generate report
            report = generate_stock_report(symbol)
            
            if report:
                sections = report.get("sections", {})
                
                # Print key information
                print(f"✅ Report generated successfully!")
                print(f"Title: {report.get('title', 'N/A')}")
                print(f"Symbol: {report.get('symbol', 'N/A')}")
                
                # Current metrics
                current = sections.get("current_metrics", {})
                print(f"Current Price: {current.get('current_price', 'N/A')}")
                print(f"Price Change: {current.get('price_change', 'N/A')}")
                print(f"Market Cap: {current.get('market_cap', 'N/A')}")
                
                # Company info
                company = sections.get("company_info", {})
                print(f"Company: {company.get('name', 'N/A')}")
                
                # Recommendation
                rec = sections.get("recommendation", {})
                print(f"Recommendation: {rec.get('action', 'N/A')} - {rec.get('confidence', 'N/A')}")
                
                # Check for errors
                if "error" in sections:
                    print(f"⚠️  Error: {sections['error']}")
                
            else:
                print("❌ Failed to generate report")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print(f"\n✅ TESTING COMPLETE!")

def test_symbol_normalization():
    """Test the symbol normalization function"""
    
    print(f"\n🔄 TESTING SYMBOL NORMALIZATION")
    print("=" * 40)
    
    test_cases = [
        ("tesla", "TSLA"),
        ("apple", "AAPL"),
        ("microsoft", "MSFT"),
        ("google", "GOOGL"),
        ("AAPL", "AAPL"),  # Already correct
        ("Amazon", "AMZN"),
        ("meta", "META"),
        ("nvidia", "NVDA")
    ]
    
    for input_symbol, expected in test_cases:
        result = normalize_stock_symbol(input_symbol)
        status = "✅" if result == expected else "❌"
        print(f"{status} {input_symbol} -> {result} (expected: {expected})")

if __name__ == "__main__":
    test_symbol_normalization()
    test_stock_reports()

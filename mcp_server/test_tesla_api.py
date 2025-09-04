#!/usr/bin/env python3
"""
Test the enhanced Flask API with real Tesla data
"""
import requests
import json
import time

def test_tesla_api():
    """Test Tesla stock report via API"""
    
    base_url = "http://localhost:5000"
    
    print("🚀 TESTING ENHANCED TESLA API")
    print("=" * 50)
    
    # Wait a moment for the server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Test health check first
    try:
        health_response = requests.get(f"{base_url}/api/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Server is running!")
        else:
            print(f"❌ Server health check failed: {health_response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running.")
        return
    
    # Test Tesla stock report
    print(f"\n📊 Testing Tesla Stock Report...")
    
    tesla_tests = [
        {"symbol": "tesla"},    # Test name -> symbol conversion
        {"symbol": "TSLA"},     # Test direct symbol
    ]
    
    for test_case in tesla_tests:
        symbol = test_case["symbol"]
        print(f"\n🔍 Testing with symbol: '{symbol}'")
        
        try:
            response = requests.post(
                f"{base_url}/api/report/generate",
                json={
                    "report_type": "stock",
                    "symbol": symbol
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    report = data.get("report", {})
                    sections = report.get("sections", {})
                    
                    print(f"✅ Report generated successfully!")
                    print(f"Generated at: {data.get('generated_at')}")
                    print(f"Title: {report.get('title')}")
                    print(f"Symbol: {report.get('symbol')}")
                    
                    # Current metrics
                    current = sections.get("current_metrics", {})
                    print(f"\n💰 CURRENT METRICS:")
                    print(f"  Price: {current.get('current_price', 'N/A')}")
                    print(f"  Change: {current.get('price_change', 'N/A')}")
                    print(f"  Market Cap: {current.get('market_cap', 'N/A')}")
                    print(f"  P/E Ratio: {current.get('pe_ratio', 'N/A')}")
                    print(f"  52-Week Range: {current.get('52_week_range', 'N/A')}")
                    
                    # Financial highlights
                    financial = sections.get("financial_highlights", {})
                    print(f"\n📈 FINANCIAL HIGHLIGHTS:")
                    print(f"  Revenue Growth: {financial.get('revenue_growth', 'N/A')}")
                    print(f"  Profit Margin: {financial.get('profit_margin', 'N/A')}")
                    print(f"  Sector: {financial.get('sector', 'N/A')}")
                    print(f"  Industry: {financial.get('industry', 'N/A')}")
                    
                    # Company info
                    company = sections.get("company_info", {})
                    print(f"\n🏢 COMPANY INFO:")
                    print(f"  Name: {company.get('name', 'N/A')}")
                    print(f"  Employees: {company.get('employees', 'N/A')}")
                    
                    # Recommendation
                    rec = sections.get("recommendation", {})
                    print(f"\n🎯 RECOMMENDATION:")
                    print(f"  Action: {rec.get('action', 'N/A')}")
                    print(f"  Target Price: ${rec.get('target_price', 'N/A')}")
                    print(f"  Confidence: {rec.get('confidence', 'N/A')}")
                    
                    # Check if we got real data
                    price = current.get('current_price', '$0.00')
                    if price != '$0.00' and 'N/A' not in price:
                        print(f"\n✅ SUCCESS: Real Tesla data fetched!")
                    else:
                        print(f"\n⚠️  WARNING: May not have real data")
                        if "error" in sections:
                            print(f"  Error: {sections['error']}")
                
                else:
                    print(f"❌ API returned error: {data}")
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    test_tesla_api()

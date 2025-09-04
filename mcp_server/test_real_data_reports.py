#!/usr/bin/env python3
"""
Test script for real data financial reports
Tests stock analysis, market summary, and portfolio analysis with live data
"""

import requests
import json
from datetime import datetime

def test_real_data_reports():
    """Test all report types with real data"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 TESTING REAL DATA FINANCIAL REPORTS")
    print("=" * 80)
    
    # Test companies to analyze
    test_companies = ["GOOGL", "AAPL", "TSLA", "MSFT", "AMZN"]
    
    for symbol in test_companies:
        print(f"\n📊 TESTING STOCK REPORT: {symbol}")
        print("-" * 50)
        
        # Test stock report generation
        response = requests.post(f"{base_url}/api/report/generate", 
                               json={
                                   "report_type": "stock",
                                   "symbol": symbol
                               })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {symbol} Stock Report Generated Successfully")
            
            # Print key metrics
            report = data.get("report", {})
            sections = report.get("sections", {})
            
            # Current metrics
            current_metrics = sections.get("current_metrics", {})
            print(f"Current Price: {current_metrics.get('current_price', 'N/A')}")
            print(f"Price Change: {current_metrics.get('price_change', 'N/A')}")
            print(f"Market Cap: {current_metrics.get('market_cap', 'N/A')}")
            print(f"P/E Ratio: {current_metrics.get('pe_ratio', 'N/A')}")
            
            # Financial highlights
            financial = sections.get("financial_highlights", {})
            print(f"Revenue Growth: {financial.get('revenue_growth', 'N/A')}")
            print(f"Sector: {financial.get('sector', 'N/A')}")
            
            # Recommendation
            recommendation = sections.get("recommendation", {})
            print(f"Recommendation: {recommendation.get('action', 'N/A')}")
            print(f"Target Price: ${recommendation.get('target_price', 'N/A')}")
            print(f"Confidence: {recommendation.get('confidence', 'N/A')}")
            
        else:
            print(f"❌ Failed to generate {symbol} report: {response.status_code}")
            print(response.text)
    
    # Test market report
    print(f"\n🌍 TESTING MARKET SUMMARY REPORT")
    print("-" * 50)
    
    response = requests.post(f"{base_url}/api/report/generate", 
                           json={"report_type": "market"})
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Market Report Generated Successfully")
        
        report = data.get("report", {})
        sections = report.get("sections", {})
        
        print(f"Market Overview: {sections.get('market_overview', 'N/A')}")
        print(f"Volatility: {sections.get('volatility_index', 'N/A')}")
        
        # Major indices
        indices = sections.get("major_indices", {})
        print("\nMajor Indices:")
        for index, value in indices.items():
            print(f"  {index}: {value}")
        
        # Sector performance
        sectors = sections.get("sector_analysis", {})
        print("\nSector Performance:")
        for sector, performance in sectors.items():
            print(f"  {sector}: {performance}")
            
    else:
        print(f"❌ Failed to generate market report: {response.status_code}")
        print(response.text)
    
    # Test portfolio report (if user has portfolio)
    print(f"\n💼 TESTING PORTFOLIO ANALYSIS REPORT")
    print("-" * 50)
    
    response = requests.post(f"{base_url}/api/report/generate", 
                           json={
                               "report_type": "portfolio",
                               "user_id": "web_user"
                           })
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Portfolio Report Generated Successfully")
        
        report = data.get("report", {})
        sections = report.get("sections", {})
        
        print(f"Overview: {sections.get('overview', 'N/A')}")
        
        # Performance metrics
        performance = sections.get("performance", {})
        if isinstance(performance, dict):
            print(f"Total Value: {performance.get('total_value', 'N/A')}")
            print(f"Total Return: {performance.get('total_return_percent', 'N/A')}")
            print(f"Positions: {performance.get('number_of_positions', 'N/A')}")
        
        # Sector allocation
        sectors = sections.get("sector_allocation", {})
        if sectors:
            print("\nSector Allocation:")
            for sector, allocation in sectors.items():
                print(f"  {sector}: {allocation}")
        
        # Recommendations
        recommendations = sections.get("recommendations", [])
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
                
    else:
        print(f"❌ Failed to generate portfolio report: {response.status_code}")
        print(response.text)
    
    print(f"\n🎉 TESTING COMPLETE!")
    print("=" * 80)

def test_comprehensive_analysis():
    """Test comprehensive analysis with real data"""
    
    print(f"\n🔍 TESTING COMPREHENSIVE ANALYSIS")
    print("-" * 50)
    
    # Test the comprehensive analysis endpoint (if available)
    base_url = "http://localhost:5000"
    
    # This would use the real data from the reports to generate analysis
    response = requests.post(f"{base_url}/api/analysis/comprehensive", 
                           json={
                               "symbol": "GOOGL",
                               "user_id": "web_user"
                           })
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Comprehensive Analysis Generated")
        # Print analysis (would be formatted nicely)
    else:
        print("ℹ️  Comprehensive analysis endpoint not available yet")

if __name__ == "__main__":
    print("🚀 REAL DATA FINANCIAL REPORTS TESTING")
    print("=" * 80)
    print("Make sure the Flask app is running on http://localhost:5000")
    print("=" * 80)
    
    try:
        test_real_data_reports()
        test_comprehensive_analysis()
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure Flask app is running")
        print("Run: python app.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

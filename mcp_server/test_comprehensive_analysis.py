import requests
import json

def test_comprehensive_analysis():
    """Test the comprehensive analysis endpoint with sample data"""
    
    # Sample stock report data
    stock_report = {
        "symbol": "AAPL",
        "current_price": 175.43,
        "price_change_percent": 2.35,
        "sector": "Technology",
        "recommendation": {
            "action": "BUY",
            "target_price": 195.00,
            "confidence": "High"
        },
        "key_metrics": {
            "pe_ratio": 28.5,
            "market_cap": 2750000000000,
            "revenue_growth": 8.2,
            "profit_margin": 25.3
        },
        "technical_analysis": {
            "overall_trend": "Bullish",
            "support_level": 170.00,
            "resistance_level": 180.00,
            "rsi": 65.2
        },
        "fundamentals": {
            "revenue": 394328000000,
            "net_income": 99803000000,
            "total_debt": 120069000000
        }
    }
    
    # Sample portfolio report data
    portfolio_report = {
        "total_value": 250000,
        "holdings": {
            "AAPL": {
                "shares": 285,
                "value": 50000,
                "weight": 20.0
            },
            "MSFT": {
                "shares": 150,
                "value": 45000,
                "weight": 18.0
            },
            "GOOGL": {
                "shares": 100,
                "value": 35000,
                "weight": 14.0
            },
            "NVDA": {
                "shares": 80,
                "value": 40000,
                "weight": 16.0
            },
            "JNJ": {
                "shares": 200,
                "value": 32000,
                "weight": 12.8
            }
        },
        "performance": {
            "total_return_percent": 12.5,
            "ytd_return": 8.7,
            "volatility": 18.2
        },
        "diversification": {
            "risk_level": "Moderate",
            "sector_count": 5,
            "correlation_score": 0.65
        }
    }
    
    # Sample market summary data
    market_summary = {
        "overall_trend": "Bullish",
        "sentiment": "Positive",
        "key_indicators": {
            "vix": 18.5,
            "sp500_change": 1.2,
            "nasdaq_change": 2.1,
            "ten_year_yield": 4.25
        },
        "sector_performance": {
            "Technology": 2.5,
            "Healthcare": 0.8,
            "Financials": 1.1,
            "Energy": -0.5
        },
        "economic_factors": {
            "gdp_growth": 2.1,
            "inflation_rate": 3.2,
            "unemployment_rate": 3.8
        }
    }
    
    # Test data
    test_data = {
        "stock_report": stock_report,
        "portfolio_report": portfolio_report,
        "market_summary": market_summary
    }
    
    try:
        print("Testing Comprehensive Analysis Endpoint...")
        print("=" * 60)
        
        url = "http://localhost:5000/api/analysis/comprehensive"
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis Generation Successful!")
            print("\n" + "="*60)
            print("COMPREHENSIVE INVESTMENT ANALYSIS")
            print("="*60)
            print(result['analysis'])
            print("="*60)
        else:
            print(f"\n❌ Analysis failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Flask app is not running. Please start the app first.")
        print("Run: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_individual_functions():
    """Test the individual analysis functions directly"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL ANALYSIS FUNCTIONS")
    print("="*60)
    
    from app import (
        generate_stock_analysis_section,
        generate_portfolio_analysis_section, 
        generate_market_integration_section,
        generate_strategic_recommendations
    )
    
    # Sample data (same as above but simplified for testing)
    stock_data = {
        "symbol": "AAPL",
        "current_price": 175.43,
        "price_change_percent": 2.35,
        "sector": "Technology",
        "recommendation": {"action": "BUY", "target_price": 195.00},
        "key_metrics": {"pe_ratio": 28.5, "market_cap": 2750000000000},
        "technical_analysis": {"overall_trend": "Bullish", "support_level": 170.00, "resistance_level": 180.00}
    }
    
    portfolio_data = {
        "total_value": 250000,
        "holdings": {"AAPL": {"value": 50000}},
        "performance": {"total_return_percent": 12.5},
        "diversification": {"risk_level": "Moderate", "sector_count": 5}
    }
    
    market_data = {
        "overall_trend": "Bullish",
        "sentiment": "Positive", 
        "key_indicators": {"vix": 18.5}
    }
    
    try:
        print("\n📊 Stock Analysis Section:")
        print("-" * 40)
        stock_section = generate_stock_analysis_section(stock_data)
        print(stock_section[:200] + "...\n")
        
        print("💼 Portfolio Analysis Section:")
        print("-" * 40)
        portfolio_section = generate_portfolio_analysis_section(portfolio_data, stock_data)
        print(portfolio_section[:200] + "...\n")
        
        print("🌍 Market Integration Section:")
        print("-" * 40)
        market_section = generate_market_integration_section(market_data, stock_data, portfolio_data)
        print(market_section[:200] + "...\n")
        
        print("🎯 Strategic Recommendations:")
        print("-" * 40)
        recommendations = generate_strategic_recommendations(stock_data, portfolio_data, market_data)
        print(recommendations[:200] + "...\n")
        
        print("✅ All individual functions working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing individual functions: {e}")

if __name__ == "__main__":
    print("COMPREHENSIVE ANALYSIS TESTING")
    print("="*60)
    
    # Test API endpoint
    test_comprehensive_analysis()
    
    # Test individual functions
    test_individual_functions()

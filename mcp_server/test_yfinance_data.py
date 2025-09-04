#!/usr/bin/env python3
"""
Test script to verify yfinance data fetching for Tesla
"""

import yfinance as yf
import json

def test_tesla_data():
    """Test fetching Tesla stock data"""
    
    print("🔍 Testing Tesla (TSLA) Data Fetching...")
    print("=" * 50)
    
    try:
        # Create ticker object
        ticker = yf.Ticker('TSLA')
        
        # Get stock info
        info = ticker.info
        
        # Get historical data
        hist = ticker.history(period="5d")
        
        print("✅ Successfully fetched Tesla data!")
        print(f"Company Name: {info.get('longName', 'N/A')}")
        print(f"Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
        print(f"Previous Close: ${info.get('previousClose', 'N/A')}")
        print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "Market Cap: N/A")
        print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        print(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}")
        
        # Check if we have recent price data
        if not hist.empty:
            latest_close = hist['Close'].iloc[-1]
            latest_volume = hist['Volume'].iloc[-1]
            print(f"Latest Close: ${latest_close:.2f}")
            print(f"Latest Volume: {latest_volume:,}")
        
        print("\n📊 Available Data Fields:")
        important_fields = [
            'longName', 'currentPrice', 'regularMarketPrice', 'previousClose',
            'marketCap', 'trailingPE', 'sector', 'industry', 'fiftyTwoWeekHigh',
            'fiftyTwoWeekLow', 'volume', 'averageVolume', 'beta', 'dividendYield'
        ]
        
        for field in important_fields:
            value = info.get(field, 'N/A')
            print(f"  {field}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error fetching Tesla data: {str(e)}")
        return False

def test_other_stocks():
    """Test other popular stocks"""
    
    print("\n🔍 Testing Other Popular Stocks...")
    print("=" * 50)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            name = info.get('longName', symbol)
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            print(f"✅ {symbol}: {name} - ${price}")
            
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

if __name__ == "__main__":
    success = test_tesla_data()
    
    if success:
        test_other_stocks()
        print(f"\n✅ Data fetching is working correctly!")
        print("The issue might be in the Flask app's symbol handling.")
    else:
        print(f"\n❌ yfinance data fetching failed!")
        print("Need to check internet connection or yfinance installation.")

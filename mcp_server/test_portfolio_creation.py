import requests
import json

def test_portfolio_creation():
    """Test the portfolio creation endpoint"""
    url = "http://localhost:5000/api/portfolio/create"
    
    test_data = {
        "name": "tech growth portfolio",
        "holdings": {
            "AAPL": 100,
            "MSFT": 50,
            "GOOGL": 25
        },
        "user_id": "test_user"
    }
    
    try:
        print("Testing portfolio creation...")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        print("=" * 50)
        
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ Portfolio creation successful!")
        else:
            print(f"\n❌ Portfolio creation failed with status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Flask app is not running. Please start the app first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_portfolio_creation()

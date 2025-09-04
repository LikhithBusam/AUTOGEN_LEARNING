import requests
import json

# Test the portfolio query through the API
url = "http://localhost:5000/api/query"
data = {
    "query": "Create a diversified portfolio recommendation for $10,000",
    "user_id": "test_user"
}

try:
    response = requests.post(url, json=data)
    result = response.json()
    
    print("Portfolio Query Test Results:")
    print("=" * 50)
    print(f"Status Code: {response.status_code}")
    print(f"Query Type: {result.get('query_type', 'NOT FOUND')}")
    print(f"Success: {result.get('success', False)}")
    
    if result.get('sentence_format'):
        print(f"\nSentence Format:")
        print(result['sentence_format'][:200] + "...")
    
    if result.get('table_format'):
        print(f"\nTable Format Preview:")
        print(result['table_format'][:300] + "...")
    
    if result.get('investment_amount'):
        print(f"\nInvestment Amount: ${result['investment_amount']:,.0f}")
    
    print(f"\nFull Response Keys: {list(result.keys())}")
        
except requests.exceptions.ConnectionError:
    print("Error: Flask app is not running. Start the app first with: python app.py")
except Exception as e:
    print(f"Error: {e}")

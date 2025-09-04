from app import IntelligentFinancialAssistant, FinancialDataMCPClient, SimpleDataAnalyst

# Initialize required dependencies
mcp_client = FinancialDataMCPClient()
data_analyst = SimpleDataAnalyst()

# Test the portfolio methods
assistant = IntelligentFinancialAssistant(mcp_client, data_analyst)

print("Testing portfolio methods existence:")
print("Has _handle_portfolio_query:", hasattr(assistant, '_handle_portfolio_query'))
print("Has _create_ai_portfolio_recommendation:", hasattr(assistant, '_create_ai_portfolio_recommendation'))
print("Has _basic_portfolio_guidance:", hasattr(assistant, '_basic_portfolio_guidance'))

print("\nTesting portfolio query detection:")
test_query = "Create a diversified portfolio recommendation for $10,000"
detected_keywords = assistant._detect_portfolio_keywords(test_query)
print("Detected keywords:", detected_keywords)
print("Is portfolio query:", assistant._is_portfolio_query(test_query))

print("\nTesting process_query method:")
# Test the full query processing flow
result = assistant.process_query(test_query, "test_user")
print("Query type:", result.get('query_type'))
print("Success:", result.get('success'))
if result.get('sentence_format'):
    print("Sentence format:", result['sentence_format'][:100] + "...")

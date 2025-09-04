import inspect
from app import IntelligentFinancialAssistant

# Check class methods
print("IntelligentFinancialAssistant methods:")
methods = [method for method in dir(IntelligentFinancialAssistant) if not method.startswith('__')]
for method in sorted(methods):
    print(f"  {method}")

print(f"\nTotal methods: {len(methods)}")

# Check specifically for portfolio methods
portfolio_methods = ['_handle_portfolio_query', '_create_ai_portfolio_recommendation', '_basic_portfolio_guidance', '_detect_portfolio_keywords', '_is_portfolio_query']
print("\nPortfolio method check:")
for method in portfolio_methods:
    exists = hasattr(IntelligentFinancialAssistant, method)
    print(f"  {method}: {exists}")

"""
Simple Flask test to debug issues
"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Flask is working!</h1><p>MCP Financial Analyst will be available soon.</p>'

@app.route('/health')
def health():
    return {'status': 'ok', 'message': 'Flask server is running'}

if __name__ == '__main__':
    print("Starting simple Flask test server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

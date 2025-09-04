#!/usr/bin/env python3
"""
Flask web interface demo for news sentiment analysis
"""
import sys
import os
sys.path.insert(0, '.')

from flask import Flask, request, jsonify, render_template_string

# Import our intelligent assistant
from app import intelligent_assistant, news_analyzer

# Create Flask app
demo_app = Flask(__name__)

# HTML template for the demo
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Financial News Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .query-section { background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .result-section { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        input[type="text"] { width: 70%; padding: 10px; margin-right: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #5a67d8; }
        .news-item { border-left: 4px solid #667eea; padding: 10px; margin: 10px 0; background: #f7f8fc; }
        .sentiment-positive { border-left-color: #10b981; }
        .sentiment-negative { border-left-color: #ef4444; }
        .sentiment-neutral { border-left-color: #6b7280; }
        .loading { color: #667eea; font-style: italic; }
        .error { color: #ef4444; }
        .success { color: #10b981; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AI Financial News Assistant</h1>
        <p>Get real-time news and sentiment analysis for any stock</p>
    </div>
    
    <div class="query-section">
        <h3>Ask about any stock's news or sentiment:</h3>
        <input type="text" id="queryInput" placeholder="e.g., 'Get latest news for Apple' or 'Tesla sentiment analysis'" value="">
        <button onclick="analyzeQuery()">Analyze</button>
        
        <div style="margin-top: 10px;">
            <strong>Example queries:</strong>
            <ul>
                <li><a href="#" onclick="setQuery('Get latest news for Apple')">Get latest news for Apple</a></li>
                <li><a href="#" onclick="setQuery('What is Tesla sentiment?')">What is Tesla sentiment?</a></li>
                <li><a href="#" onclick="setQuery('Microsoft news analysis')">Microsoft news analysis</a></li>
                <li><a href="#" onclick="setQuery('Market news today')">Market news today</a></li>
            </ul>
        </div>
    </div>
    
    <div id="results" class="result-section" style="display: none;">
        <h3>Analysis Results</h3>
        <div id="resultContent"></div>
    </div>

    <script>
        function setQuery(query) {
            document.getElementById('queryInput').value = query;
        }
        
        function analyzeQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query.trim()) {
                alert('Please enter a query');
                return;
            }
            
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultContent');
            
            resultsDiv.style.display = 'block';
            contentDiv.innerHTML = '<div class="loading">🔄 Analyzing... This may take a moment.</div>';
            
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                contentDiv.innerHTML = '<div class="error">❌ Error: ' + error.message + '</div>';
            });
        }
        
        function displayResults(data) {
            const contentDiv = document.getElementById('resultContent');
            
            if (!data.success) {
                contentDiv.innerHTML = '<div class="error">❌ Analysis failed: ' + (data.error || 'Unknown error') + '</div>';
                return;
            }
            
            let html = '';
            
            // AI Response
            if (data.sentence_format) {
                html += '<div class="success"><h4>🤖 AI Analysis:</h4><p>' + data.sentence_format + '</p></div>';
            }
            
            // Query Type
            html += '<div><strong>Query Type:</strong> ' + (data.query_type || 'Unknown') + '</div>';
            
            // News Data
            if (data.analysis && data.analysis.news_data) {
                const newsData = data.analysis.news_data;
                
                if (newsData.overall_sentiment) {
                    const sentiment = newsData.overall_sentiment;
                    const sentimentClass = 'sentiment-' + sentiment.label;
                    html += '<div class="news-item ' + sentimentClass + '">';
                    html += '<h4>📊 Overall Sentiment: ' + sentiment.label.toUpperCase() + ' (' + sentiment.score.toFixed(2) + ')</h4>';
                    html += '</div>';
                }
                
                if (newsData.articles && newsData.articles.length > 0) {
                    html += '<h4>📰 News Articles (' + newsData.articles.length + ' found):</h4>';
                    
                    newsData.articles.forEach((article, index) => {
                        const articleSentiment = article.sentiment || {label: 'neutral', score: 0};
                        const sentimentClass = 'sentiment-' + articleSentiment.label;
                        
                        html += '<div class="news-item ' + sentimentClass + '">';
                        html += '<strong>' + (article.title || 'No title') + '</strong><br>';
                        html += '<small>Source: ' + (article.source || 'Unknown') + ' | ';
                        html += 'Sentiment: ' + articleSentiment.label + ' (' + articleSentiment.score.toFixed(2) + ')</small>';
                        if (article.summary) {
                            html += '<p>' + article.summary + '</p>';
                        }
                        html += '</div>';
                    });
                }
            }
            
            contentDiv.innerHTML = html;
        }
        
        // Allow Enter key to submit
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeQuery();
            }
        });
    </script>
</body>
</html>
"""

@demo_app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@demo_app.route('/api/analyze', methods=['POST'])
def analyze_query():
    """Analyze a financial query"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        # Process the query with our intelligent assistant
        result = intelligent_assistant.process_query(query)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@demo_app.route('/api/news/<symbol>')
def get_stock_news(symbol):
    """Get news for a specific stock symbol"""
    try:
        limit = request.args.get('limit', 5, type=int)
        result = news_analyzer.get_stock_news(symbol, limit=limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@demo_app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_enabled': intelligent_assistant is not None,
        'news_analyzer_available': news_analyzer is not None
    })

if __name__ == '__main__':
    print("🚀 Starting AI Financial News Assistant Demo")
    print("📱 Open your browser to: http://localhost:5000")
    print("🤖 AI-powered news sentiment analysis ready!")
    
    demo_app.run(debug=True, host='0.0.0.0', port=5000)

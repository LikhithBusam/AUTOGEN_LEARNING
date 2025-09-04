"""
Simple HTTP server for testing
"""
import http.server
import socketserver
import json
from urllib.parse import urlparse, parse_qs

class FinancialAnalystHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>MCP Financial Analyst - Test Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                    h1 { color: #333; text-align: center; }
                    .status { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin: 20px 0; }
                    .info { background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 15px; margin: 20px 0; }
                    .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }
                    .test-form { margin: 20px 0; }
                    .test-form input { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
                    .test-form button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 MCP-Powered Financial Analyst</h1>
                    <div class="status">
                        <strong>✅ Test Server Running!</strong><br>
                        The server is working correctly. Flask application is being prepared.
                    </div>
                    
                    <div class="info">
                        <h3>System Status</h3>
                        <ul>
                            <li>✅ HTTP Server: Running on port 8000</li>
                            <li>🔄 Flask Application: Being prepared</li>
                            <li>🤖 AutoGen Agents: Ready for initialization</li>
                            <li>📊 Financial APIs: Configured</li>
                        </ul>
                    </div>
                    
                    <h3>Available Endpoints</h3>
                    <div class="endpoint"><strong>GET /</strong> - This dashboard</div>
                    <div class="endpoint"><strong>GET /health</strong> - Health check</div>
                    <div class="endpoint"><strong>POST /api/test</strong> - Test endpoint</div>
                    
                    <div class="test-form">
                        <h3>Test Query</h3>
                        <form action="/api/test" method="post">
                            <input type="text" name="query" placeholder="Ask about stocks..." style="width: 300px;">
                            <button type="submit">Submit</button>
                        </form>
                    </div>
                    
                    <div class="info">
                        <h3>Next Steps</h3>
                        <p>The full Flask application with AutoGen agents will be available shortly. This test server confirms that:</p>
                        <ul>
                            <li>Network connectivity is working</li>
                            <li>Python HTTP server is functional</li>
                            <li>Web interface can be served</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                "status": "healthy",
                "server": "test_http_server",
                "message": "Test server is running",
                "next_step": "Flask application preparation"
            }
            self.wfile.write(json.dumps(health_data, indent=2).encode())
            
    def do_POST(self):
        if self.path == '/api/test':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "success": True,
                "message": "Test endpoint working",
                "received_data": post_data.decode(),
                "server": "test_http_server"
            }
            self.wfile.write(json.dumps(response, indent=2).encode())

if __name__ == '__main__':
    PORT = 8000
    print(f"Starting test server on http://localhost:{PORT}")
    print("This confirms network connectivity and basic web server functionality")
    print("Press Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", PORT), FinancialAnalystHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}/")
        httpd.serve_forever()

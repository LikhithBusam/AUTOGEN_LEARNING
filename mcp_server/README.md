# MCP-Powered Financial Analyst with AutoGen

A sophisticated multi-agent financial analysis system powered by AutoGen and Model Context Protocol (MCP).

## 🚀 Features

- **Multi-Agent System**: 6 specialized agents working together
- **Real-time Financial Data**: Integration with multiple financial APIs via MCP
- **Sentiment Analysis**: News and social media sentiment analysis
- **Visual Reports**: Automated chart generation and PDF reports
- **Portfolio Optimization**: Monte Carlo simulations and Modern Portfolio Theory
- **Natural Language Queries**: Ask complex financial questions in plain English
- **Memory & Persistence**: Remembers user preferences and portfolio data

## 🏗️ Architecture

### Agents
- **Orchestrator Agent**: Manages workflow and coordinates other agents
- **Data Analyst Agent**: Retrieves and analyzes financial data via MCP
- **News & Sentiment Agent**: Gathers news and performs sentiment analysis
- **Report Generator Agent**: Creates structured PDF/HTML reports
- **Visualization Agent**: Generates charts and graphs
- **Recommendation Agent**: Provides investment suggestions and risk analysis

### MCP Integrations
- Alpha Vantage API
- Yahoo Finance
- News APIs
- Social Media APIs

## 📁 Project Structure

```
mcp_server/
├── agents/                 # AutoGen agent implementations
├── mcp/                   # MCP server and client implementations
├── reports/               # Generated reports output
├── data/                  # Database and data storage
├── config/                # Configuration files
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## 📊 Example Queries

- "Compare Tesla and Ford earnings this quarter"
- "What's the sentiment around Apple stock this week?"
- "Generate a risk analysis for my tech portfolio"
- "Show me candlestick charts for Amazon over the last 3 months"
- "What if inflation rises to 7%? (scenario analysis)"

## 🔧 Configuration

Edit `config/settings.py` to configure:
- API keys
- Database settings
- Agent parameters
- Report templates

## 📈 Advanced Features

- Portfolio optimization with Modern Portfolio Theory
- Monte Carlo simulations for risk analysis
- SEC filings summarization
- Voice interface with Whisper + TTS
- Real-time price alerts
- Scenario analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see LICENSE file for details

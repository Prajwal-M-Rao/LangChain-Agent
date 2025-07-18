
# Enhanced LangChain Wikipedia Agent

This project implements an advanced interactive question-answering agent using **LangChain**, a local **LLM (TinyLlama-1.1B-Chat-v1.0)**, and multiple knowledge sources including **Wikipedia**, **DuckDuckGo**, and **Wolfram Alpha**. The agent follows the **ReAct (Reasoning + Acting) framework** with enhanced decision-making capabilities, confidence scoring, and intelligent source selection.

## 🚀 Key Features

### Core Capabilities
- **Multi-Source Knowledge Integration**: Wikipedia, DuckDuckGo web search, and Wolfram Alpha for computational queries
- **Intelligent Source Selection**: Agent dynamically chooses the most appropriate knowledge source based on query type
- **Confidence-Based Decision Making**: Implements confidence thresholds to optimize external API usage
- **Query Classification**: Automatically categorizes queries (factual, computational, current events, etc.)
- **Context-Aware Responses**: Maintains conversation history and provides contextually relevant answers
- **Fallback Mechanisms**: Graceful handling of API failures with automatic source switching

### Advanced Features
- **Response Caching**: Reduces API calls by caching recent responses
- **Query Similarity Detection**: Avoids redundant searches for similar questions
- **Conversation Memory**: Maintains context across multiple exchanges
- **Performance Metrics**: Tracks response times, API usage, and confidence scores
- **Extensible Architecture**: Easy to add new knowledge sources and tools

## 📋 Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **Internet connection** for external API access

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd enhanced-wikipedia-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Local LLM
The TinyLlama model will be automatically downloaded on first run. Ensure you have ~2GB of free disk space.

### 5. Configuration (Optional)
Create a `.env` file for API keys:
```env
WOLFRAM_ALPHA_API_KEY=your_wolfram_alpha_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
```

## 🚀 Usage

### Basic Usage
```bash
python main.py
```

### Interactive Mode
```bash
python main.py --interactive
```

### Configuration Options
```bash
python main.py --config config.yaml --log-level DEBUG
```

### Example Usage Session
```
🤖 Enhanced Wikipedia Agent initialized!
📚 Available sources: Wikipedia, DuckDuckGo, Wolfram Alpha
💡 Type 'help' for commands, 'exit' to quit

You: What is the capital of France?
🔍 Query classified as: FACTUAL
💭 Confidence: 0.95 (High) - Using existing knowledge
🤖 Agent: The capital of France is Paris. Paris is the largest city in France and serves as the country's political, economic, and cultural center.

You: Calculate the square root of 256
🔍 Query classified as: COMPUTATIONAL  
🔧 Using Wolfram Alpha for mathematical computation
🤖 Agent: The square root of 256 is 16.

You: What are the latest developments in AI research?
🔍 Query classified as: CURRENT_EVENTS
🌐 Using DuckDuckGo for recent information
🤖 Agent: Based on recent search results, here are the latest developments in AI research...
```

## 🎯 Expected Behavior

### Intelligent Decision Making
- **High Confidence (>0.8)**: Agent responds directly using existing knowledge
- **Medium Confidence (0.5-0.8)**: Agent may consult Wikipedia for verification
- **Low Confidence (<0.5)**: Agent automatically searches external sources

### Source Selection Logic
- **Factual Questions**: Wikipedia (historical facts, biographical info, scientific concepts)
- **Current Events**: DuckDuckGo web search (news, recent developments)
- **Computational Queries**: Wolfram Alpha (mathematical calculations, conversions)
- **Mixed Queries**: Multi-source approach with result synthesis

### Response Quality Features
- **Source Attribution**: All external information includes proper citations
- **Confidence Indicators**: Each response includes confidence level
- **Response Caching**: Similar queries return cached results for speed
- **Error Recovery**: Graceful fallback when primary sources fail

## 🔧 Configuration

### Configuration File (config.yaml)
```yaml
llm:
  model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  temperature: 0.7
  max_tokens: 512

agent:
  confidence_threshold: 0.6
  max_search_results: 5
  enable_caching: true
  cache_duration: 3600  # 1 hour

sources:
  wikipedia:
    enabled: true
    top_k_results: 3
    lang: "en"
  
  duckduckgo:
    enabled: true
    max_results: 5
    region: "us-en"
  
  wolfram_alpha:
    enabled: true
    timeout: 30

logging:
  level: "INFO"
  file: "agent.log"
  console: true
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Installation Issues
```bash
# If you encounter dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### Model Loading Problems
```bash
# Clear Hugging Face cache and retry
rm -rf ~/.cache/huggingface/
python main.py
```

#### API Connection Issues
- **Wikipedia**: Check internet connection and try reducing `top_k_results`
- **DuckDuckGo**: Verify no VPN/proxy issues blocking requests
- **Wolfram Alpha**: Ensure API key is valid and within rate limits

#### Performance Issues
- **Slow Responses**: Enable caching and reduce `max_search_results`
- **Memory Usage**: Decrease `max_tokens` in configuration
- **API Rate Limits**: Increase delays between requests in config

### Error Codes
- **E001**: LLM initialization failed
- **E002**: Wikipedia API unavailable
- **E003**: Invalid configuration file
- **E004**: Insufficient system resources

## 📊 Performance Monitoring

### Built-in Metrics
- Response time tracking
- API usage statistics
- Confidence score distributions
- Cache hit/miss ratios

### Monitoring Dashboard
```bash
python monitor.py --dashboard
```

## 🔮 Future Enhancements

### Planned Features
- **Vector Database Integration**: Semantic search capabilities with ChromaDB
- **Multi-Language Support**: Queries and responses in multiple languages
- **Custom Knowledge Base**: Upload and integrate domain-specific documents
- **Advanced Reasoning**: Chain-of-thought prompting for complex queries
- **Voice Interface**: Speech-to-text and text-to-speech capabilities
- **Web UI**: Browser-based interface for enhanced user experience

### Advanced Integrations
- **Arxiv Integration**: Access to academic papers and research
- **News API**: Real-time news and current events
- **Google Scholar**: Academic and scientific information
- **Social Media APIs**: Trending topics and public sentiment

### Architecture Improvements
- **Microservices**: Distributed architecture for scalability
- **Database Backend**: Persistent storage for conversation history
- **API Gateway**: RESTful API for external integrations
- **Container Support**: Docker deployment for easy distribution

## 🤝 Contributing

### Development Setup
```bash
git clone <repository_url>
cd enhanced-wikipedia-agent
pip install -r requirements-dev.txt
pre-commit install
```

### Adding New Knowledge Sources
1. Create a new tool class in `src/tools/`
2. Implement the required interface methods
3. Add configuration options in `config.yaml`
4. Update the agent's tool selection logic

### Testing
```bash
pytest tests/
python -m pytest tests/ --cov=src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful agent framework
- **Hugging Face**: For the TinyLlama model
- **Wikipedia**: For providing free access to knowledge
- **Wolfram Alpha**: For computational intelligence
- **DuckDuckGo**: For privacy-focused search capabilities

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and ideas
- **Email**: [your-email@example.com]

---

**Version**: 2.0.0  
**Last Updated**: July 2025  
**Compatibility**: Python 3.8+, LangChain 0.1.0+

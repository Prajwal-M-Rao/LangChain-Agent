#!/usr/bin/env python3
"""
Enhanced Wikipedia Agent - Main Entry Point
A sophisticated question-answering agent using LangChain, local LLM, and multiple knowledge sources.
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import utilities
from utils.config_loader import config, create_directories, validate_config
from utils.logger import logger, log_info, log_error, log_warning
from utils.metrics import metrics, export_report

# Import core components (these will be implemented next)
# from agents.enhanced_react_agent import EnhancedReactAgent
# from llm.local_llm import LocalLLMManager


def setup_environment():
    """Setup the environment for the agent."""
    try:
        # Create necessary directories
        create_directories()
        
        # Validate configuration
        if not validate_config():
            log_error("Configuration validation failed")
            return False
            
        log_info("Environment setup completed successfully")
        return True
    except Exception as e:
        log_error(f"Environment setup failed: {e}")
        return False


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("🤖 Enhanced Wikipedia Agent")
    print("=" * 60)
    print("📚 Available sources: Wikipedia, DuckDuckGo, Wolfram Alpha")
    print("💡 Type 'help' for commands, 'exit' to quit")
    print("🔧 Configuration loaded successfully")
    print("=" * 60)


def print_help():
    """Print help information."""
    help_text = """
Available Commands:
  help        - Show this help message
  exit, quit  - Exit the agent
  stats       - Show performance statistics
  config      - Show current configuration
  reset       - Reset conversation history
  sources     - Show available knowledge sources
  debug       - Toggle debug mode
  
Query Types:
  📖 Factual: "What is the capital of France?"
  🧮 Math: "Calculate the square root of 256"
  📰 Current: "Latest news about AI"
  💬 Chat: "Hello, how are you?"
  
Examples:
  "Tell me about quantum computing"
  "What happened in 2024?"
  "Solve: 2x + 5 = 15"
  "Compare Python and JavaScript"
    """
    print(help_text)


def interactive_mode():
    """Run the agent in interactive mode."""
    print_welcome()
    
    # TODO: Initialize agent
    # agent = EnhancedReactAgent()
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤔 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'stats':
                show_stats()
                continue
            elif user_input.lower() == 'config':
                show_config()
                continue
            elif user_input.lower() == 'reset':
                print("🔄 Conversation history reset")
                continue
            elif user_input.lower() == 'sources':
                show_sources()
                continue
            elif user_input.lower() == 'debug':
                toggle_debug()
                continue
                
            # Process query
            print("🤖 Agent: [Agent implementation pending]")
            print("Query received:", user_input)
            
            # TODO: Process with agent
            # response = agent.process_query(user_input)
            # print(f"🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            log_error(f"Error in interactive mode: {e}")
            print(f"❌ Error: {e}")


def show_stats():
    """Show performance statistics."""
    try:
        report = metrics.get_performance_report()
        
        print("\n📊 Performance Statistics")
        print("-" * 30)
        
        # System metrics
        system = report.get("system", {})
        print(f"🖥️  CPU Usage: {system.get('cpu_percent', 0):.1f}%")
        print(f"💾 Memory Usage: {system.get('memory_percent', 0):.1f}%")
        print(f"💿 Disk Usage: {system.get('disk_percent', 0):.1f}%")
        
        # Counters
        counters = report.get("counters", {})
        if counters:
            print(f"📈 Total Queries: {counters.get('queries_total', 0)}")
            print(f"⚠️  Total Errors: {counters.get('errors_total', 0)}")
            
        # Derived metrics
        derived = report.get("derived_metrics", {})
        if derived:
            print(f"📉 Error Rate: {derived.get('error_rate_percent', 0):.2f}%")
            
    except Exception as e:
        log_error(f"Error showing stats: {e}")
        print(f"❌ Error retrieving stats: {e}")


def show_config():
    """Show current configuration."""
    print("\n⚙️  Current Configuration")
    print("-" * 30)
    
    # LLM config
    llm_config = config.get_section("llm")
    print(f"🧠 Model: {llm_config.get('model_name', 'Not set')}")
    print(f"🌡️  Temperature: {llm_config.get('temperature', 'Not set')}")
    print(f"📏 Max Tokens: {llm_config.get('max_tokens', 'Not set')}")
    
    # Agent config
    agent_config = config.get_section("agent")
    print(f"🎯 Confidence Threshold: {agent_config.get('confidence_threshold', 'Not set')}")
    print(f"💾 Caching: {'Enabled' if agent_config.get('enable_caching') else 'Disabled'}")
    
    # Sources
    sources_config = config.get_section("sources")
    print(f"📚 Wikipedia: {'Enabled' if sources_config.get('wikipedia', {}).get('enabled') else 'Disabled'}")
    print(f"🔍 DuckDuckGo: {'Enabled' if sources_config.get('duckduckgo', {}).get('enabled') else 'Disabled'}")
    print(f"🧮 Wolfram Alpha: {'Enabled' if sources_config.get('wolfram_alpha', {}).get('enabled') else 'Disabled'}")


def show_sources():
    """Show available knowledge sources."""
    print("\n📚 Available Knowledge Sources")
    print("-" * 30)
    
    sources_config = config.get_section("sources")
    
    # Wikipedia
    if sources_config.get("wikipedia", {}).get("enabled", False):
        print("✅ Wikipedia - Factual information, biographies, concepts")
    else:
        print("❌ Wikipedia - Disabled")
        
    # DuckDuckGo
    if sources_config.get("duckduckgo", {}).get("enabled", False):
        print("✅ DuckDuckGo - Current events, web search")
    else:
        print("❌ DuckDuckGo - Disabled")
        
    # Wolfram Alpha
    if sources_config.get("wolfram_alpha", {}).get("enabled", False):
        print("✅ Wolfram Alpha - Mathematical computations")
    else:
        print("❌ Wolfram Alpha - Disabled")


def toggle_debug():
    """Toggle debug mode."""
    global debug_mode
    debug_mode = not getattr(toggle_debug, 'debug_mode', False)
    toggle_debug.debug_mode = debug_mode
    
    if debug_mode:
        print("🐛 Debug mode enabled")
        logger.setLevel("DEBUG")
    else:
        print("🐛 Debug mode disabled")
        logger.setLevel("INFO")


def single_query_mode(query: str):
    """Process a single query and exit."""
    try:
        if not setup_environment():
            return 1
            
        log_info(f"Processing single query: {query}")
        
        # TODO: Initialize agent
        # agent = EnhancedReactAgent()
        # response = agent.process_query(query)
        # print(response)
        
        print(f"[Single query mode] Query: {query}")
        print("[Agent implementation pending]")
        
        return 0
        
    except Exception as e:
        log_error(f"Error in single query mode: {e}")
        print(f"❌ Error: {e}")
        return 1


def batch_mode(input_file: str, output_file: str = None):
    """Process queries from a file in batch mode."""
    try:
        if not setup_environment():
            return 1
            
        input_path = Path(input_file)
        if not input_path.exists():
            log_error(f"Input file not found: {input_file}")
            return 1
            
        # Read queries from file
        with open(input_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
            
        if not queries:
            log_warning("No queries found in input file")
            return 0
            
        log_info(f"Processing {len(queries)} queries from {input_file}")
        
        # TODO: Initialize agent
        # agent = EnhancedReactAgent()
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n🔄 Processing query {i}/{len(queries)}: {query}")
            
            try:
                # TODO: Process with agent
                # response = agent.process_query(query)
                response = f"[Batch mode] Query {i}: {query} - [Agent implementation pending]"
                results.append({
                    'query': query,
                    'response': response,
                    'success': True
                })
                
            except Exception as e:
                log_error(f"Error processing query {i}: {e}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
                
        # Save results
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Query: {result['query']}\n")
                    if result['success']:
                        f.write(f"Response: {result['response']}\n")
                    else:
                        f.write(f"Error: {result['error']}\n")
                    f.write("-" * 50 + "\n")
            
            log_info(f"Results saved to {output_file}")
            
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 Batch Processing Summary:")
        print(f"✅ Successful: {successful}/{len(queries)}")
        print(f"❌ Failed: {len(queries) - successful}/{len(queries)}")
        
        return 0
        
    except Exception as e:
        log_error(f"Error in batch mode: {e}")
        return 1


def benchmark_mode():
    """Run benchmark tests."""
    try:
        if not setup_environment():
            return 1
            
        log_info("Starting benchmark mode")
        
        # Test queries for benchmarking
        test_queries = [
            "What is the capital of France?",
            "Calculate the square root of 256",
            "Tell me about quantum computing",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
            "Solve: x^2 + 5x + 6 = 0",
            "What is machine learning?",
            "When was the Internet invented?",
            "What is the speed of light?"
        ]
        
        print(f"🏃 Running benchmark with {len(test_queries)} queries...")
        
        # TODO: Initialize agent
        # agent = EnhancedReactAgent()
        
        start_time = asyncio.get_event_loop().time()
        
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Query {i}/{len(test_queries)}: {query}")
            
            query_start = asyncio.get_event_loop().time()
            
            try:
                # TODO: Process with agent
                # response = agent.process_query(query)
                response = f"[Benchmark] Query {i}: {query} - [Agent implementation pending]"
                
                query_time = asyncio.get_event_loop().time() - query_start
                print(f"⏱️  Response time: {query_time:.2f}s")
                
            except Exception as e:
                log_error(f"Error in benchmark query {i}: {e}")
                
        total_time = asyncio.get_event_loop().time() - start_time
        avg_time = total_time / len(test_queries)
        
        print(f"\n📈 Benchmark Results:")
        print(f"🕐 Total Time: {total_time:.2f}s")
        print(f"⏱️  Average Time: {avg_time:.2f}s per query")
        print(f"🚀 Throughput: {len(test_queries)/total_time:.2f} queries/second")
        
        # Export detailed metrics
        export_report()
        
        return 0
        
    except Exception as e:
        log_error(f"Error in benchmark mode: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Wikipedia Agent - Intelligent Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py -q "What is quantum computing?"    # Single query
  python main.py -b queries.txt -o results.txt     # Batch mode
  python main.py --benchmark                        # Benchmark mode
  python main.py --test                            # Test mode
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query to process and exit'
    )
    
    parser.add_argument(
        '-b', '--batch',
        type=str,
        help='Input file for batch processing'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for batch results'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark tests'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test suite'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Enhanced Wikipedia Agent v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        toggle_debug()
    
    # Load custom config if provided
    if args.config:
        config.load_config(args.config)
    
    try:
        # Route to appropriate mode
        if args.query:
            return single_query_mode(args.query)
        elif args.batch:
            return batch_mode(args.batch, args.output)
        elif args.benchmark:
            return benchmark_mode()
        elif args.test:
            return run_tests()
        else:
            # Default to interactive mode
            if not setup_environment():
                return 1
            interactive_mode()
            return 0
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return 1


def run_tests():
    """Run the test suite."""
    try:
        print("🧪 Running test suite...")
        
        # Test environment setup
        print("🔧 Testing environment setup...")
        if not setup_environment():
            print("❌ Environment setup failed")
            return 1
        print("✅ Environment setup passed")
        
        # Test configuration
        print("⚙️  Testing configuration...")
        if not validate_config():
            print("❌ Configuration validation failed")
            return 1
        print("✅ Configuration validation passed")
        
        # Test utilities
        print("🔧 Testing utilities...")
        try:
            # Test logger
            log_info("Test log message")
            
            # Test metrics
            metrics.increment_counter("test_counter")
            report = metrics.get_performance_report()
            
            print("✅ Utilities test passed")
        except Exception as e:
            print(f"❌ Utilities test failed: {e}")
            return 1
        
        # TODO: Test agent components when implemented
        print("🤖 Testing agent components...")
        print("⏳ Agent tests pending implementation")
        
        print("\n🎉 All tests passed!")
        return 0
        
    except Exception as e:
        log_error(f"Error running tests: {e}")
        print(f"❌ Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
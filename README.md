# LangChain Wikipedia Agent

This project implements an interactive question-answering agent using **LangChain**, a local **LLM (TinyLlama-1.1B-Chat-v1.0)**, and the **Wikipedia API**. The agent follows the **ReAct (Reasoning + Acting) framework**, dynamically determining whether external information from Wikipedia is required based on user queries.

## Prerequisites

Ensure you have the following installed:

- **Python 3.8+**
- **pip** (Python package manager)

## Installation

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```
Usage

Steps to Use

1.Run the script.Enter a question when prompted.
2.The agent determines whether it can answer directly or needs additional information from Wikipedia.1.
3.If required, the agent fetches relevant data from Wikipedia and formulates a response.
4.Type 'exit' to terminate the program.

Expected Behavior

How the Agent Responds

If the agent has sufficient knowledge, it responds directly.
If additional factual support is required, it retrieves and integrates data from Wikipedia before responding.

Troubleshooting

Common Issues and Fixes

Ensure all dependencies are installed correctly.
If Wikipedia queries fail, check your internet connection.
If API errors occur, try reducing top_k_results in WikipediaAPIWrapper.

Future Enhancements

Potential Improvements

Implement confidence thresholds to optimize Wikipedia usage.
Integrate additional knowledge sources for broader coverage.
Improve error handling for API requests.

Conclusion

This project demonstrates an intelligent conversational agent that leverages LangChain, a local LLM, and the Wikipedia API for enhanced question-answering capabilities.

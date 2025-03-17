LangChain Wikipedia Agent
This project implements an interactive question-answering agent using LangChain, a local LLM (TinyLlama-1.1B-Chat-v1.0), and the Wikipedia API. The agent follows the ReAct (Reasoning + Acting) framework, dynamically determining whether external information from Wikipedia is required based on user queries.

Prerequisites
Ensure you have the following installed:

Python 3.8+
pip (Python package manager)
Installation
Clone the repository:

bash
Copy
Edit
git clone <repository_url>
cd <repository_folder>
Install dependencies:

bash
Copy
Edit
pip install langchain transformers huggingface_hub wikipedia-api
Running the Project
To start the LangChain Wikipedia Agent, run:

bash
Copy
Edit
python main.py
Usage
Run the script.
Enter a question when prompted.
The agent determines whether it can answer directly or needs additional information from Wikipedia.
If required, the agent fetches relevant data from Wikipedia and formulates a response.
Type 'exit' to terminate the program.
Expected Behavior
If the agent has sufficient knowledge, it responds directly.
If additional factual support is required, it retrieves and integrates data from Wikipedia before responding.
Troubleshooting
Ensure all dependencies are installed correctly.
If Wikipedia queries fail, check your internet connection.
If API errors occur, try reducing top_k_results in WikipediaAPIWrapper.
Future Enhancements
Implement confidence thresholds to optimize Wikipedia usage.
Integrate additional knowledge sources for broader coverage.
Improve error handling for API requests.
This project demonstrates an intelligent conversational agent that leverages LangChain, a local LLM, and the Wikipedia API for enhanced question-answering capabilities.

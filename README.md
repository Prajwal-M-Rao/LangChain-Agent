# LangChain-Agent
This task involves creating a conversational agent using the LangChain framework along with an open-source language model (LLM). The agent should be capable of holding a conversation with a user and provide helpful responses, potentially utilizing external tools to fetch real-time information to enhance the responses.
Project Overview

This project implements a question-answering agent using LangChain, a local LLM (TinyLlama-1.1B-Chat-v1.0), and Wikipedia API integration. The agent dynamically determines whether to retrieve external information from Wikipedia based on user queries, following the ReAct (Reasoning + Acting) framework.

Prerequisites

Ensure you have the following installed:

Python 3.8+

pip package manager

Installation

Clone the repository:

git clone <repository_url>
cd <repository_folder>

Install required dependencies:

pip install langchain transformers huggingface_hub wikipedia-api

Running the Project

To start the LangChain Wikipedia Agent, execute the following command:

python main.py

Usage

Run the script.

Enter a question when prompted.

The agent will determine whether it can answer directly or needs additional information from Wikipedia.

If needed, Wikipedia data will be fetched and included in the response.

Type 'exit' to terminate the program.

Expected Output

If the agent has sufficient knowledge, it responds directly.

If additional factual support is needed, it retrieves data from Wikipedia before formulating the answer.

Troubleshooting

Ensure all dependencies are installed correctly.

If Wikipedia queries fail, check your internet connection.

If API errors occur, try reducing top_k_results in WikipediaAPIWrapper.

Future Improvements

Add confidence thresholds to refine Wikipedia usage.

Integrate additional knowledge sources.

Enhance error handling for API requests.

This project demonstrates an interactive question-answering system leveraging LangChain, a local LLM, and Wikipedia API.


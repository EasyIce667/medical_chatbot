# Medical Chatbot

A Streamlit-based medical chatbot application that leverages LangChain, HuggingFace embeddings, and FAISS vector stores to provide answers based on medical document context.

## Overview

This project implements a medical chatbot that uses natural language processing to answer user queries by retrieving relevant information from a vectorized database of medical documents. The application is built using Python, Streamlit, and integrates with HuggingFace models for language understanding.

## About Medical Chatbot

### What is Medical Chatbot?

The Medical Chatbot is an innovative application designed to assist users by providing informed responses to medical-related queries. Built using Python, Streamlit, and advanced natural language processing techniques, this project leverages the power of LangChain, HuggingFace embeddings, and FAISS vector stores to create a knowledgeable and interactive chatbot.

### Purpose and Functionality

This project aims to simplify access to medical information by allowing users to ask questions in natural language. The chatbot retrieves relevant data from a pre-processed database of medical documents, offering detailed answers and citing source references when available. It is particularly useful for educational purposes, quick reference, and understanding medical concepts without requiring deep expertise.

#### Key Features:

- **Natural Language Querying**: Users can type questions in plain English to get responses.
- **Document-Based Knowledge**: Powered by a vectorized database of PDF medical documents.
- **Source Attribution**: Provides references to source documents for transparency.
- **User-Friendly Interface**: Built with Streamlit for an intuitive web-based experience.

### How It Works

1. **Data Preparation**: Medical documents in PDF format are loaded and split into manageable chunks.
2. **Vectorization**: The text chunks are converted into embeddings using HuggingFace's sentence-transformers and stored in a FAISS index.
3. **Query Processing**: User queries are processed by a language model (e.g., LLaMA via Groq API) that retrieves relevant information from the vector store.
4. **Response Generation**: The system generates a detailed response based on the context and displays it in the Streamlit interface.

### Target Audience

- Medical students seeking quick references.
- Healthcare professionals needing a tool for preliminary information.
- General users interested in learning about medical topics.

### Development and Future Scope

Developed as an open-source project, the Medical Chatbot is a work in progress. Future enhancements may include support for more languages, integration with real-time medical databases, and improved accuracy through advanced machine learning models.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/EasyIce667/medical-chatbot.git
   cd medical-chatbot
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Create a `.env` file in the root directory.
   - Add your HuggingFace token and GROQ API key:
     ```
     HF_TOKEN=your_huggingface_token
     GROQ_API_KEY=your_groq_api_key
     ```

4. Prepare the vector database:
   - Place your PDF medical documents in the `data/` directory.
   - Run the `create_mem_llm.py` script to generate the FAISS vector store:
     ```bash
     python create_mem_llm.py
     ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run medibot.py
   ```

2. Open your browser and navigate to `http://localhost:xxxx`.
3. Enter your medical-related queries in the chat interface and explore the responses.

## Files

- `medibot.py`: Main application file using Streamlit and LangChain.
- `create_mem_llm.py`: Script to load PDFs and create the FAISS vector store.
- `connect_mem_to_llm.py`: Example of connecting the LLM with the FAISS database.
- `requirements.txt`: List of dependencies required for the project.

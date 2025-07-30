# Agentic RAG Chatbot

This project implements an agentic Retrieval-Augmented Generation (RAG) chatbot. The application allows users to upload documents and ask questions about their content. The backend is built using the Multi-Agent Communication Protocol (MCP) to orchestrate a set of specialized agents that handle document ingestion, retrieval, and response generation. The user interface is created with Streamlit.

## Architecture

The system follows a client-server architecture orchestrated by MCP. The Streamlit application acts as the client, sending requests to an MCP server that manages a collection of agents. Each agent is responsible for a specific task in the RAG pipeline.

The agents in the system are:

* **StreamlitApp:** The user-facing interface for file upload and chat.
* **MCP Client:** Resides within the Streamlit app and communicates with the MCP Server.
* **MCP Server:** The central hub that routes messages between the agents.
* **IngestionAgent:** Responsible for loading and chunking the uploaded documents.
* **DocumentLoader:** A utility class used by the IngestionAgent to handle various document formats (PDF, TXT, DOCX, etc.).
* **RetrievalAgent:** Manages the creation of embeddings and the FAISS vector store. It handles both storing document chunks and retrieving relevant context for a given query.
* **LLMResponseAgent:** Takes the user's query and the retrieved context to generate a final answer using a Google Gemini LLM.

## Project Flow

The overall process can be broken down into two main pipelines: the Ingestion Pipeline and the Query & Response Pipeline.

### 1. Ingestion Pipeline

1.  **User Uploads Document:** The process begins when the user uploads a document through the Streamlit UI.
2.  **IngestionAgent:** The `IngestionAgent` receives the file path. It uses the `DocumentLoader` to load the document's content and then splits the text into smaller, manageable chunks.
3.  **RetrievalAgent:** The `RetrievalAgent` takes these chunks, generates vector embeddings for each using a Google embedding model, and stores them in a FAISS vector database for efficient similarity search.

### 2. Query & Response Pipeline

1.  **User Asks Question:** The user submits a question through the chat interface.
2.  **RetrievalAgent:** The `RetrievalAgent` receives the query, creates an embedding for it, and searches the FAISS vector database to find the most relevant document chunks (context).
3.  **LLMResponseAgent:** This agent receives the original query and the retrieved context. It constructs a prompt that includes both and sends it to the Google Gemini LLM.
4.  **Google Gemini LLM:** The LLM generates a final, human-readable answer based on the provided context and question.
5.  **Display Answer:** The final answer, along with the source chunks, is sent back to the Streamlit UI and displayed to the user.

## File Descriptions

* `app.py`: The main Streamlit application. It handles the user interface, file uploads, and communication with the MCP server.
* `server.py`: The MCP server that defines and exposes the agentic tools (`ingest_and_store_document`, `answer_question`) for the client.
* `IngestionAgent.py`: Contains the `IngestionAgent` class, which is responsible for loading and chunking documents.
* `DocumentLoader.py`: A helper class for loading various document types.
* `RetrievalAgent.py`: Manages the FAISS vector store, including adding documents and retrieving relevant chunks.
* `LLMResponseAgent.py`: Responsible for generating the final answer using the LLM and the retrieved context.
* `requirements.txt`: A list of all the Python dependencies required to run the project.
* `Dockerfile`: A file used to build a Docker image for easy deployment of the application.

## How to Run

### Prerequisites

* Python 3.10+
* An environment variable `GOOGLE_API_KEY` set with a valid Google API key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AkshithSai-24/Agentic-RAG-MCP.git
    cd Agentic-RAG-MCP
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### Usage

1.  Upload a document using the file uploader in the sidebar.
2.  Once the document is ingested, you can ask questions about its content in the chat input box.
3.  The chatbot will provide an answer and the sources it used to generate the response.

## Docker Deployment

You can also run this application within a Docker container.

1.  **Build the Docker image:**
    ```bash
    docker build -t rag-chatbot .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 -e GOOGLE_API_KEY="YOUR_API_KEY" rag-chatbot
    ```

Now you can access the application at `http://localhost:8501`.

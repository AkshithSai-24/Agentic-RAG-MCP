from mcp.server.fastmcp import FastMCP
from RetrivalAgent import RetrievalAgent
from IngestionAgent import IngestionAgent
from LLMResponseAgent import LLMResponseAgent   
from typing import List, Dict, Any
import json
import os
import shutil
import logging



mcp_server = FastMCP("RAG_Agent_Server",port=9898)


vector_store = RetrievalAgent()

def create_response_message(request_data: Dict[str, Any], sender: str, type: str, payload: Dict[str, Any]) -> str:
    """Helper function to construct a JSON response message."""
    response = {
        "sender": sender,
        "receiver": request_data.get("sender"),
        "type": type,
        "trace_id": request_data.get("trace_id"),
        "payload": payload
    }
    return json.dumps(response)



@mcp_server.tool()
def ingest_and_store_document(message: str) -> str:
    """
    Loads a document from the given file path, splits it into chunks,
    creates embeddings, and stores them in the vector database.
    This tool prepares documents for the question-answering tool.
    """

    request_data = json.loads(message)

    file_path = request_data["payload"]["file_path"]

    print(file_path,flush=True)
    print(f"[Tool: ingest_and_store_document] Processing {file_path}")
    
    # Ingestion Logic

    try:

        ingestion_agent = IngestionAgent()

        chunks =  ingestion_agent.run(file_path) # <--- This uses file_path

        if not chunks:
            print(f"[Tool: ingest_and_store_document] Failure: No chunks were created from {file_path}.")
            payload = {"status": "failure", "message": f"Could not process the document. It might be empty, corrupted, or an unsupported format."}
            print(payload)
            return create_response_message(request_data, "IngestionAgent", "INGESTION_RESPONSE", payload)
        

    except Exception as e:
        # The error "name 'file_path' is not defined" is occurring within this block.
        payload = {"status": "failure", "message": f"An error occurred: {e}"}
        return create_response_message(request_data, "IngestionAgent", "INGESTION_RESPONSE", payload)
    try:
        vector_store.add_documents(chunks)
        payload = {"status": "success", "message": f"Successfully ingested {len(chunks)} chunks from {file_path}."}
        return create_response_message(request_data, "RetrivalAgent", "INGESTION_RESPONSE", payload)
    except Exception as e:
        payload = {"status": "failure", "message": f"An error occurred while adding documents: {e}"}
        return create_response_message(request_data, "RetrivalAgent", "INGESTION_RESPONSE", payload)





# --- Tool 2: Answer Question ---
@mcp_server.tool()
def answer_question(message: str) -> str:
    """
    Answers a user's question by retrieving relevant context from the
    ingested documents and using an LLM to generate a final response.
    """
    request_data = json.loads(message)
    query = request_data["payload"]["query"]
    print(f"[Tool: answer_question] Answering query: '{query}'")


    # Response Generation Logic
    retriever = vector_store.get_retriever()

    llm_agent = LLMResponseAgent(retriever=retriever)
    
    try:
        result = llm_agent.run(query)
        source_docs = [doc.page_content for doc in result.get("source_documents", [])]
        payload = {
            "status": "success",
            "result": result.get("result"),
            "source_chunks": source_docs
        }
        return create_response_message(request_data, "LLMResponseAgent", "QA_RESPONSE", payload)
    except Exception as e:
        payload = {"status": "failure", "result": f"An error occurred: {e}"}
        return create_response_message(request_data, "LLMResponseAgent", "QA_RESPONSE", payload)


if __name__ == "__main__":
    print("Starting RAG MCP Server...")

    mcp_server.run(transport="stdio")


import os
import streamlit as st
import json
import uuid
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import shutil

def cleanup_old_files():
    print("Running startup cleanup...")
    # Delete the uploads directory
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
        print("Cleaned up old 'uploads' directory.")
    # Delete the FAISS vector store
    if os.path.exists("faiss_vector_store"):
        shutil.rmtree("faiss_vector_store")
        print("Cleaned up old 'faiss_vector_store' directory.")


# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Chatbot")
st.caption("Upload a document and ask questions about its content. Powered by LangChain and MCP.")
cleanup_old_files()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None

# --- Helper Functions ---
async def run_rag_pipeline(file_path: str, query: str):
    """
    Connects to the MCP server and runs the full RAG pipeline.
    This function is adapted from the async client logic.
    """
    

    server_params = StdioServerParameters(command="python", args=["./server.py"])
    agent_name = "StreamlitUI"
    trace_id = str(uuid.uuid4())
    
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                # 1. Ingestion
                ingest_request = {
                    "sender": agent_name, "receiver": "IngestionAgent", "type": "INGESTION_REQUEST",
                    "trace_id": trace_id, "payload": {"file_path": os.path.abspath(file_path)}
                }
                st.info("Step 1: Ingesting document...")
                ingest_result = await session.call_tool("ingest_and_store_document", arguments={"message": json.dumps(ingest_request)})
                ingest_response = json.loads(ingest_result.content[0].text)
                if ingest_response["payload"]["status"] == "failure":
                    st.error(f"Ingestion failed: {ingest_response['payload']['message']}")
                    return None

                # 2. Question Answering
                qa_request = {
                    "sender": agent_name, "receiver": "LLMResponseAgent", "type": "QA_REQUEST",
                    "trace_id": trace_id, "payload": {"query": query}
                }
                st.info("Step 2: Retrieving context and generating answer...")
                qa_result = await session.call_tool("answer_question", arguments={"message": json.dumps(qa_request)})
                qa_response = json.loads(qa_result.content[0].text)
                st.info("Step 3: Complete!")
                return qa_response["payload"]
    except Exception as e:
        st.error(f"An error occurred in the backend pipeline: {e}")
        return None

# --- UI Components ---

# Sidebar for file upload
with st.sidebar:
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, DOCX, or PPTX file",
        type=["pdf", "txt", "md", "docx", "pptx", "csv"]
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Akshith Sai Kondamadugu")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = file_path
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")




# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.code(source, language="text")

if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if a file has been uploaded
    if st.session_state.uploaded_file_path is None:
        st.warning("Please upload a document first.", icon="⚠️")
    else:
        # Process the request with the RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the async function using asyncio.run()
                response_payload = asyncio.run(run_rag_pipeline(st.session_state.uploaded_file_path, prompt))
                
                if response_payload and response_payload.get("status") == "success":
                    answer = response_payload.get("result", "No answer found.")
                    sources = response_payload.get("source_chunks", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                             for source in sources:
                                st.code(source, language='text')
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_message = "Sorry, I couldn't process that request. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

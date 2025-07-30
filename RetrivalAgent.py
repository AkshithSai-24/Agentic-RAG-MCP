import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class RetrievalAgent:

    def __init__(self, persist_directory: str = "./faiss_vector_store"):
        print("[RetrievalAgent] Initializing...") # ADD THIS
        self.persist_directory = persist_directory
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            print("[RetrievalAgent] Embeddings initialized successfully.") # ADD THIS
        except Exception as e:
            print(f"[RetrievalAgent] ERROR: Failed to initialize embeddings: {e}") # ADD THIS
            raise # Re-raise to crash early if critical
        self.vector_store = self._load_or_create_store()
        print("[RetrievalAgent] Initialization complete.") # ADD THIS

    def _load_or_create_store(self) -> FAISS:
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"Loading FAISS store from {self.persist_directory}")
            return FAISS.load_local(
                self.persist_directory, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            print("Creating new FAISS store.")
            dummy_texts = ["initialization"]
            initial_store = FAISS.from_texts(dummy_texts, self.embeddings)
            initial_store.save_local(self.persist_directory)
            return initial_store

    def add_documents(self, documents):
        if not documents:
            return
        print(f"[RetrievalAgent] Adding {len(documents)} chunks to FAISS.")
        self.vector_store.add_documents(documents)
        self.vector_store.save_local(self.persist_directory)
        print("[RetrievalAgent] Vector store updated.")

    def get_retriever(self, search_kwargs={"k": 5}):
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
if __name__ == "__main__":  
    # Example usage
    agent = RetrievalAgent()
    test_documents = [Document(page_content="This is a test document.", metadata={"source": "test.txt"})]
    agent.add_documents(test_documents)
    
    retriever = agent.get_retriever()
    results = retriever.get_relevant_documents("What is a test?")
    
    for doc in results:
        print(f"Content: {doc.page_content}, Source: {doc.metadata}")
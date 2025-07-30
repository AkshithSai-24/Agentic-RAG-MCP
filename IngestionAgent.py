from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from DocumentLoader import DocumentLoader
load_dotenv()


class IngestionAgent:
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.loader = DocumentLoader()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def run(self, file_path: str) :
        
        print(f"[IngestionAgent] Starting ingestion for: {file_path}")
        
        documents = self.loader.load(file_path)
        
        if not documents:
            print(f"[IngestionAgent] Failed to load or file is empty: {file_path}")
            return []

        split_chunks = self.text_splitter.split_documents(documents)
        
        print(f"[IngestionAgent] Successfully created {len(split_chunks)} chunks from {file_path}")
        return split_chunks
if __name__ == "__main__":
    # Example usage
    agent = IngestionAgent()
    file_path = "test.pptx"  # Replace with your file path
    chunks = agent.run(file_path)
    
    if chunks:
        print(f"Successfully created {len(chunks)} chunks from {file_path}.")
    else:
        print(f"Failed to create chunks from {file_path}.")
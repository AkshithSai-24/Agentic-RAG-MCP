from dotenv import load_dotenv
import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)

load_dotenv()

class DocumentLoader:

    def load(self, file_path: str) :

        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        
        print(f"[Document Loader] Loading '{os.path.basename(file_path)}' with LangChain...")

        try:
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            elif extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif extension == '.pptx':
                loader = UnstructuredPowerPointLoader(file_path)
            elif extension == '.csv':
                loader = CSVLoader(file_path)
            else:
                print(f"Unsupported file format: {extension}")
                return []
            
            return loader.load()
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    file_path = "test.pptx"  # Replace with your file path
    documents = loader.load(file_path)
    
    if documents:
        print(f"Successfully loaded {len(documents)} documents from {file_path}.")
    else:
        print(f"Failed to load documents from {file_path}.")
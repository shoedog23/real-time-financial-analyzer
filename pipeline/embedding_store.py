from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks (List[langchain.schema.Document]): The document chunks to embed and store.

    Returns:
        FAISS: The FAISS vector store containing the embeddings.
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create a FAISS vector store from the document chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

# Example usage
if __name__ == "__main__":
    from pdf_processing import process_pdf
    
    # Process a sample PDF to generate chunks
    chunks = process_pdf("/Users/natesh/Downloads/Projects/real-time-financial-analyzer/data/filings/aapl-20200926.pdf")
    
    # Create the FAISS vector store
    vector_store = create_vector_store(chunks)
    
    print("FAISS vector store created successfully!")
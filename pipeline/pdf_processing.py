from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Process and chunk a single PDF file
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

# Example usage:
if __name__ == "__main__":
    chunks = process_pdf("/Users/natesh/Downloads/Projects/real-time-financial-analyzer/data/filings/aapl-20200926.pdf")
    print(chunks[:10])
    print(f"Number of chunks: {len(chunks)}")
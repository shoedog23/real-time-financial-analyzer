from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

def create_rag_pipeline(vector_store):
    """
    Create a Retrieval-Augmented Generation (RAG) pipeline using GPT-4o-mini.

    Args:
        vector_store (FAISS): The vector store for retrieval.

    Returns:
        RetrievalQAWithSourcesChain: The RAG pipeline.
    """
    # Define a prompt template for financial analysis tasks
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a financial analyst specializing in FAANG companies. You write in a very concise way, similar to Wall Street Journal articles.
        Question: {question}
        Context: {context}
        Provide a response based on the provided context.
        """
    )

    # Initialize GPT-4o-mini LLM with model parameters
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # Set low temperature for factual responses
        max_tokens=500,  # Limit token usage for concise answers
        openai_api_key=openai_api_key  # Pass API key explicitly
    )

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    # Create the RetrievalQAWithSourcesChain using `from_chain_type`
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template, "document_variable_name": "context"},  # Pass the prompt template here
        return_source_documents=True  # Return source documents along with the result
    )
    
    return qa_chain


# Example usage of RAG pipeline
if __name__ == "__main__":
    from pdf_processing import process_pdf
    from embedding_store import create_vector_store

    # Process a sample PDF to generate chunks
    chunks = process_pdf("/Users/natesh/Downloads/Projects/real-time-financial-analyzer/data/filings/aapl-20200926.pdf")

    # Create a FAISS vector store from document chunks
    vector_store = create_vector_store(chunks)

    # Create the RAG pipeline using GPT-4o-mini
    rag_pipeline = create_rag_pipeline(vector_store)

    # Define a query for testing
    query = "Give me 3 reasons why Apple is better than Google, in terms of financial metrics?"

    try:
        # Use `.invoke()` to run the query through the RAG pipeline and get results
        response = rag_pipeline.invoke({"question": query})

        # Extract and print the result and source documents
        print("Generated Answer:")
        print(response["answer"])
        
        # print("\nSource Documents:")
        # for doc in response["source_documents"]:
        #     print(f"- {doc.page_content}")
    
    except Exception as e:
        print(f"Error occurred: {e}")

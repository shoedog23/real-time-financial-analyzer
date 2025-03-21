import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from pipeline.pdf_processing import process_pdf
from pipeline.embedding_store import create_vector_store
from pipeline.rag_pipeline import create_rag_pipeline

# Set Streamlit page configuration
st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
)

# Sidebar branding
st.sidebar.title("📊 Financial Report Analyzer")
st.sidebar.markdown("Analyze 10-K filings and financial reports in real-time using AI-powered insights.")

# Main title
st.title("📈 Real-Time Financial Report Analyzer")
st.markdown("""
Welcome to the **Financial Report Analyzer**! This tool allows you to upload financial reports (e.g., 10-K filings), analyze them, and get AI-powered answers to your queries based on the uploaded documents. 🚀
""")

# File upload section
st.subheader("Step 1: Upload Financial Reports")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files containing financial reports.",
    type="pdf",
    accept_multiple_files=True,
)

# Initialize global variables for vector store and pipeline
vector_store = None
rag_pipeline = None

if uploaded_files:
    st.info("Processing uploaded files...")

    # Initialize an empty list to store all chunks from uploaded PDFs
    chunks_list = []

    # Process each uploaded PDF file
    for uploaded_file in uploaded_files:
        # Save the file temporarily
        file_path = f"temp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF and extract chunks
        chunks = process_pdf(file_path)
        chunks_list.extend(chunks)

    # Create a vector store using FAISS
    st.info("Creating vector store...")
    vector_store = create_vector_store(chunks_list)

    st.success("Files processed and vector store created successfully!")

    # Create the RAG pipeline using RetrievalQAWithSourcesChain
    st.info("Initializing RAG pipeline...")
    rag_pipeline = create_rag_pipeline(vector_store)

    st.success("RAG pipeline initialized successfully!")

# Query section
st.subheader("Step 2: Ask Questions")
query_input = st.text_input(
    "Enter your query below (e.g., 'What are Apple's key risk factors?'):",
)

if query_input and rag_pipeline:
    st.info("Processing your query...")
    
    try:
        response = rag_pipeline.invoke({"question": query_input})

        # Display the generated answer and source documents in columns for better layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Generated Answer:")
            st.markdown(f"**{response['answer']}**")

        with col2:
            st.subheader("Source Documents:")
            if response["source_documents"]:
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"Document {i + 1}:")
                    st.write(doc.page_content)
            else:
                st.warning("No relevant documents found.")
                
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Footer branding
st.markdown("---")
st.markdown("""
Developed by Natesh | Powered by LangChain & OpenAI | © 2025  
""")

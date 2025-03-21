import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from pipeline.pdf_processing import process_pdf
from pipeline.embedding_store import create_vector_store
from pipeline.rag_pipeline import create_rag_pipeline

# Title of the Streamlit app
st.title("Real-Time Financial Report Analyzer")

# File upload section
uploaded_files = st.file_uploader("Upload Financial Reports (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("Processing uploaded files...")

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
    st.write("Creating vector store...")
    vector_store = create_vector_store(chunks_list)

    st.success("Files processed and vector store created successfully!")

    # Create the RAG pipeline
    st.write("Initializing RAG pipeline...")
    qa_pipeline = create_rag_pipeline(vector_store)

    st.success("RAG pipeline initialized successfully!")

    # Query section
    query_input = st.text_input("Enter your query:")
    if query_input:
        st.write("Processing your query...")
        try:
            # Retrieve relevant documents before passing them to the RAG pipeline
            retriever = vector_store.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(query_input)

            # Print retrieved documents for debugging purposes
            st.subheader("Retrieved Documents:")
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"Document {i + 1}:")
                    st.write(doc.page_content)
            else:
                st.warning("No relevant documents found.")

            # Run the query through the RAG pipeline
            response = qa_pipeline({"question": query_input})

            # Display the generated answer and source documents
            st.subheader("Generated Answer:")
            st.write(response["answer"])

        except Exception as e:
            st.error(f"Error occurred: {e}")

import streamlit as st
import PyPDF2
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama  # Import Ollama LLM
from langchain.document_loaders import PyPDFLoader

# Set page configuration for better layout
st.set_page_config(page_title="PDF QA with Ollama", layout="wide")

st.title("📄 PDF Upload and RAG Demo using Ollama")

# Sidebar options for user customization
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
embedding_model = st.sidebar.text_input("Embedding Model", value="all-MiniLM-L6-v2")
ollama_model = st.sidebar.text_input("Ollama Model", value="llama3.2:1b")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner('Processing PDF...'):
        try:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Use PyPDFLoader for better PDF parsing
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            texts = text_splitter.split_documents(pages)

            # Create embeddings using a local model
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            docsearch = FAISS.from_documents(texts, embeddings)

            # Initialize the RAG chain with Ollama LLM
            retriever = docsearch.as_retriever()
            llm = Ollama(model=ollama_model)  # Specify the model you have in Ollama
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
            )

            st.success("✅ PDF content has been processed. You can now ask questions based on the PDF.")
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)

    # Input for user question
    user_question = st.text_input("Ask a question about the PDF content:")

    if user_question:
        with st.spinner('Generating answer...'):
            try:
                # Get the answer from the RAG model
                answer = qa_chain.run(user_question)
                st.write("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"🚨 An error occurred while generating the answer: {e}")
else:
    st.info("👆 Please upload a PDF file to begin.")

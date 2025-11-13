import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

st.title("Company Policy Q&A Bot")

# Load all .txt files from the folder
texts = []
for file in os.listdir("."):
    if file.endswith(".txt"):
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())

if not texts:
    st.write("No .txt policy files found. Add them to this folder!")
else:
    # Split text into chunks and create embeddings
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Fast, free model
    db = FAISS.from_documents(docs, embeddings)

    # User asks a question
    question = st.text_input("Ask a policy question:")
    if question:
        results = db.similarity_search(question, k=1)
        if results:
            st.success(f"**Answer:**\n\n{results[0].page_content}")
        else:
            st.error("No matching policy found. Try rephrasing!")
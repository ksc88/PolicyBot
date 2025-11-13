import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

st.title("Company Policy Q&A Bot")

# Load all .txt policy files from the folder
texts = []
for file in os.listdir("."):
    if file.endswith(".txt"):
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())

if not texts:
    st.write("❌ No .txt policy files found! Add them to this folder (e.g., vacation_policy.txt).")
else:
    # Split text into chunks for better search
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)
    
    # Use a fast, free embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build FAISS index (lightweight, no errors!)
    db = FAISS.from_documents(docs, embeddings)

    # User question input
    question = st.text_input("💬 Ask a policy question (e.g., 'How many vacation days?'):")
    
    if question:
        results = db.similarity_search(question, k=1)
        if results:
            st.success(f"**Answer:**\n\n{results[0].page_content}")
            st.info("💡 Tip: Add more .txt files for better answers!")
        else:
            st.warning("🤔 No exact match found. Try rephrasing your question!")
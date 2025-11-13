import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

# Title
st.title("Company Policy Q&A Bot")

# Load policy files
policy_folder = "."
texts = []
for file in os.listdir(policy_folder):
    if file.endswith(".txt"):
        with open(os.path.join(policy_folder, file), "r", encoding="utf-8") as f:
            texts.append(f.read())

if not texts:
    st.write("No policy files found. Add .txt files to this folder.")
else:
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts)

    # Create vector store
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding)

    # Question input
    question = st.text_input("Ask a policy question:")

    if question:
        # Search
        results = vectordb.similarity_search(question, k=1)
        if results:
            st.success(f"**Answer:** {results[0].page_content}")
        else:
            st.error("No answer found.")
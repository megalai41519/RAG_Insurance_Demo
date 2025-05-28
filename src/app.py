# src/app.py

import streamlit as st
from dotenv import load_dotenv
import os
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
device =-1

# Load environment variables
load_dotenv()

# Title
st.set_page_config(page_title="Insurance RAG Assistant", layout="wide")
st.title("🛡️ Insurance Policy Assistant (RAG-Powered)")

# Prompt input
user_question = st.text_input("💬 Ask a question about the insurance policy:")

# Set up retriever
persist_directory = './chroma_db'
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = Chroma(persist_directory=persist_directory, embedding_function=embedding_model).as_retriever(search_kwargs={"k": 3})

# Set up local LLM (HuggingFace)
hf_model_name = "google/flan-t5-small"

local_pipeline = pipeline("text2text-generation",model=hf_model_name,device="cpu")
llm = HuggingFacePipeline(pipeline=local_pipeline,model_kwargs={"max_new_tokens":128, "do_sample":True, "temperature":0.1})

# Custom prompt
prompt_template = """Answer the insurance query precisely using the provided context.

Context:
{context}

Question:
{question}

Answer (brief and accurate and provide details about the source document chunks):"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# On submit
if user_question:
    with st.spinner("🔍 Retrieving answer..."):
        result = qa_chain.invoke({"query": user_question})
        answer = result["result"]
        sources = result["source_documents"]

        st.markdown("### 🧠 Answer:")
        st.success(answer)

        st.markdown("### 📄 Retrieved Context:")
        for i, doc in enumerate(sources,1):
            snippet = doc.page_content.replace("\n", " ")[:200]
            st.markdown(f"**Chunk {i}:** {snippet}")

            

# src/ingest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Paths
PDF_PATH = "data/sample_policy.pdf"
CHROMA_DB_DIR = "data/chroma_db"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(pages)

# 3. Create embeddings (no API key needed)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 4. Store in Chroma
db = Chroma.from_documents(docs, embedding, persist_directory=CHROMA_DB_DIR)

print(f"✅ Ingested and stored {len(docs)} chunks into ChromaDB at {CHROMA_DB_DIR}")

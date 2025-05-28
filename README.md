
# 🛡️ RAG Insurance Demo

An end-to-end **Retrieval-Augmented Generation (RAG)** solution using open-source tools — designed to answer insurance policy-related questions using your own documents, embeddings, and models.

This project uses **ChromaDB**, **Hugging Face Transformers**, and **LangChain**, and runs fully **offline using `flan-t5-small`** on your local machine.

---

## 🚀 Features

- 📄 Ingests insurance policy PDFs
- 🔍 Performs semantic search via `ChromaDB`
- 🧠 Embeds text using `sentence-transformers` (`all-MiniLM-L6-v2`)
- 🤖 Uses Hugging Face’s `flan-t5-small` model via local `transformers.pipeline`
- 💬 Streamlit UI for interactive querying

---

## 🏗️ Project Structure

```
RAG_Insurance_Demo/
├── data/
│   └── sample_policy.pdf         # Sample insurance document
├── src/
│   ├── ingest.py                 # Load, chunk & store docs in ChromaDB
│   ├── embed.py                  # LLM and RetrievalQA chain
│   ├── retriever.py              # CLI tool to test question answering
│   └── app.py                    # Streamlit web app
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/RAG_Insurance_Demo.git
cd RAG_Insurance_Demo
```

### 2. Create a virtual environment and activate

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest documents

```bash
python src/ingest.py
```

### 5. Launch the Streamlit app

```bash
streamlit run src/app.py
```

---

## ❓ Example Questions

- "What is the deductible for water damage?"
- "Is flood damage covered under this policy?"
- "How soon must a claim be filed?"

---

## 💡 Local LLM Model Info

- Model used: `google/flan-t5-small`
- Loaded via: `transformers.pipeline`
- No API keys or internet required after initial download

---

## 🧠 Tech Stack

- LangChain
- ChromaDB
- Hugging Face Transformers
- Sentence Transformers
- Streamlit
- PyMuPDF

---

## 🧪 Future Enhancements

- [ ] PDF upload feature in UI  
- [ ] Dockerization for deployment  
- [ ] Switch to quantized model for speedup  
- [ ] Add LangChain Agents for multi-hop questions

---

## 📝 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

Built with ❤️ by Mani — leveraging the power of GenAI + RAG + real-world insurance knowledge.

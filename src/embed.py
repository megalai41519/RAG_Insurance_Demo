#!/usr/bin/env python3
import os
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
#from langchain_community.llms import HuggingFacePipeline
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

def main():
   
    #hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Path where ingest.py persisted your Chroma DB
    CHROMA_DIR = "data/chroma_db"

    # ——— Embedding model ————————————————————————————————————
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ——— Load your vector store —————————————————————————
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    from transformers import pipeline
    # ——— LLM setup via HF Inference API ————————————————————
    hf_model_name = "google/flan-t5-base"

    local_pipeline = pipeline("text2text-generation",model=hf_model_name,device="cpu")
    llm = HuggingFacePipeline(
    pipeline=local_pipeline,
    model_kwargs={"max_new_tokens":10, "do_sample":False, "temperature":0.0}
)
    
    prompt_template = """Given the insurance policy context below, answer ONLY "Yes", "No", or "Cannot determine from given context."'

    Context:
    {context}
    Question:
    {question}

    Answer (brief and concise):"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # ——— Build a QA chain —————————————————————————————
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    # ——— Interactive loop ——————————————————————————————
    print("Ask about your insurance policy (type 'exit' to quit)\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            result = qa.invoke(query)
            # result is a dict with 'result' and 'source_documents'
            print("\n📄 Answer:")
            print(result["result"])
            print("\n🔍 Sources:")
            for i, doc in enumerate(result["source_documents"], 1):
                snippet = doc.page_content.replace("\n", " ")[:200]
                print(f"  {i}. {snippet}...")
            print("\n" + "-"*50 + "\n")

        except Exception as e:
            print("⚠️ Error during inference:", e)
            print()

if __name__ == "__main__":
    main()

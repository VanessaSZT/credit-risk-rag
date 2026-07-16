import os
# FIX: Redirect HuggingFace traffic to a reliable mirror to bypass timeouts/firewalls
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def main():
    print("--- Starting HK Banking RAG Pipeline (Multi-Document / 100% Local) ---")

    # ==========================================
    # 1. Document Ingestion (Multi-PDF Directory)
    # ==========================================
    # Note: data/ path is relative to the execution directory
    data_dir = "data"
    loader = PyPDFDirectoryLoader(data_dir)
    pdf_docs = loader.load()
    print(f"[*] Loaded {len(pdf_docs)} page(s) across all PDFs in the 'data' directory.")

    # ==========================================
    # 2. Chunking Strategy for PDFs
    # ==========================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(pdf_docs)
    print(f"[*] Semantically split into {len(chunks)} chunks, preserving Page & Source metadata.")

    # ==========================================
    # 3. LOCAL Embedding Generation & Vector Storage
    # ==========================================
    print("[*] Loading completely local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # In-memory Chroma instance to bypass SQLite file-locking issues
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hk_credit_risk_policies"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    print("[*] Embeddings generated and stored in-memory.")

    # ==========================================
    # 4. Strict LLM Prompt & LOCAL LLM (Ollama)
    # ==========================================
    prompt_template = """You are an expert underwriter assistant for a Hong Kong bank. 
You must ONLY use the retrieved context to answer the question. 
If the context does not contain the answer, you must state 'The policy does not specify this.' 
Do not use outside knowledge.

You must explicitly list EVERY policy violation found across ALL provided documents. 
Read the context carefully regarding limits, tenors, and PEP (Politically Exposed Persons) requirements.

When answering, you MUST cite your sources strictly using the metadata provided in the context blocks below (e.g., "According to [SME_Unsecured_Lending_Guidelines.pdf - Page 1]...").

Context:
{context}

Question:
{question}

Strict Answer with Citations:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    local_model = os.environ.get("LOCAL_MODEL_NAME", "llama3.1")
    print(f"[*] Initializing local generation LLM via Ollama (Model: {local_model})...")
    llm = Ollama(model=local_model, temperature=0.0)

    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            page_num = doc.metadata.get("page", "Unknown")
            source_file = os.path.basename(doc.metadata.get("source", "Unknown_Document"))
            human_page = int(page_num) + 1 if isinstance(page_num, int) else page_num
            
            formatted_chunks.append(f"[Source Metadata: {source_file} - Page {human_page}]\n{doc.page_content}")
        return "\n\n".join(formatted_chunks)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ==========================================
    # 5. Complex Cross-Document Query Execution
    # ==========================================
    query = "We are onboarding a standard risk corporate client. They have three owners holding 30%, 20%, and 50% stakes respectively. Which individuals must have their identities verified?"
    print(f"\n[*] Querying the 100% Local RAG System:\n    \"{query}\"")
    
    retrieved_docs = retriever.invoke(query)
    print("\n[*] Retrieved Context Snippets (from local ChromaDB):")
    print(format_docs(retrieved_docs))

    print(f"\n[*] Generated Assessment (from local Ollama - {local_model}):")
    response = rag_chain.invoke(query)
    print(f"\n{response}\n")

if __name__ == "__main__":
    main()

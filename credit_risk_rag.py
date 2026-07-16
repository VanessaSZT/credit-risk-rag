"""HK Banking Credit-Risk RAG pipeline — 100% local.

Stack: PyPDF ingestion -> RecursiveCharacterTextSplitter -> HuggingFace
embeddings (all-MiniLM-L6-v2) -> in-memory ChromaDB -> Ollama generation.

Usage:
    python credit_risk_rag.py                        # run the built-in demo query
    python credit_risk_rag.py "your question here"   # ad-hoc query
    python credit_risk_rag.py --interactive          # REPL over the policy corpus
    python credit_risk_rag.py --show-context "..."   # also print retrieved chunks
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

# If huggingface.co is blocked in your region/network, opt into a mirror before
# running, e.g.:  HF_ENDPOINT=https://hf-mirror.com python credit_risk_rag.py
# (Not defaulted here: mirrors can serve stale metadata and break downloads for
# everyone else.)

# Keep the "nothing leaves this machine" promise: disable Chroma telemetry
# before chromadb is imported (its Settings honors this env var).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

DEFAULT_MODEL = os.environ.get("LOCAL_MODEL_NAME", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEMO_QUERY = (
    "A client wants an unsecured SME loan of HKD 8 million for 72 months. "
    "One of their directors is a Politically Exposed Person (PEP). "
    "What are the policy violations here according to the SME and AML guidelines?"
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Shared grounding rules so both routes behave identically on faithfulness.
GROUNDING_RULES = """You are an expert underwriter assistant for a Hong Kong bank.
You must ONLY use the retrieved context to answer the question.
If the context does not contain the answer, state exactly: 'The policy does not specify this.'
Do not use outside knowledge.
When answering, you MUST cite sources strictly using the metadata provided in the
context blocks (e.g., "According to [SME_Unsecured_Lending_Guidelines.pdf - Page 1]...")."""

COMPLIANCE_PROMPT = GROUNDING_RULES + """

The user is describing a concrete lending/onboarding scenario. You must explicitly
list EVERY policy violation found across ALL provided documents. Read the context
carefully regarding limits, tenors, thresholds, and PEP (Politically Exposed
Persons) requirements. If a stated fact complies with policy, do not list it as
a violation. End with a bulleted "Policy Violations:" summary.

Context:
{context}

Question:
{question}

Strict Answer with Citations:"""

INFORMATIONAL_PROMPT = GROUNDING_RULES + """

The user is asking an informational question about the policies. Answer it
directly and concisely. Do NOT append a policy-violation analysis — none was
requested.

Context:
{context}

Question:
{question}

Strict Answer with Citations:"""

# Lightweight LLM-based intent router (roadmap item #1 from the README).
INTENT_PROMPT = """Classify the banking query below into exactly one category:

- COMPLIANCE: the user describes a concrete deal/client scenario and wants it
  checked against policy (violations, approvals, eligibility of a specific case).
- INFORMATIONAL: the user asks what a policy says (limits, definitions,
  thresholds, procedures) without presenting a specific case to adjudicate.

Query: {question}

Respond with a single word, COMPLIANCE or INFORMATIONAL, and nothing else."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    query: str
    intent: str           # "compliance" | "informational"
    context: str          # formatted chunks exactly as shown to the LLM
    answer: str


def format_docs(docs) -> str:
    """Render retrieved chunks with the citation metadata the prompt requires."""
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get("page", "Unknown")
        source_file = os.path.basename(doc.metadata.get("source", "Unknown_Document"))
        human_page = page_num + 1 if isinstance(page_num, int) else page_num
        formatted.append(f"[Source Metadata: {source_file} - Page {human_page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def check_ollama(model: str, base_url: str = OLLAMA_BASE_URL) -> None:
    """Fail fast (before the slow embedding step) if Ollama is unreachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"[!] Cannot reach Ollama at {base_url} ({exc}).\n"
            f"    Install it from https://ollama.com and start it, e.g.: ollama run {model}"
        )
    available = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
    if model.split(":")[0] not in available:
        sys.exit(
            f"[!] Model '{model}' is not available in Ollama (found: {sorted(available) or 'none'}).\n"
            f"    Pull it first: ollama pull {model}"
        )


class CreditRiskRAG:
    """End-to-end local RAG pipeline over the PDF policy corpus."""

    def __init__(self, data_dir: str = "data", model: str = DEFAULT_MODEL, k: int = 5):
        if not os.path.isdir(data_dir) or not any(
            f.lower().endswith(".pdf") for f in os.listdir(data_dir)
        ):
            sys.exit(
                f"[!] No PDFs found in '{data_dir}/'. "
                "Generate the mock policy corpus first: python generate_mock_data.py"
            )
        check_ollama(model)

        # 1. Ingestion — every PDF page in the directory, metadata preserved.
        pdf_docs = PyPDFDirectoryLoader(data_dir).load()
        print(f"[*] Loaded {len(pdf_docs)} page(s) across all PDFs in '{data_dir}/'.")

        # 2. Chunking — small chunks with overlap so numeric limits stay intact.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pdf_docs)
        print(f"[*] Split into {len(chunks)} chunks, preserving Page & Source metadata.")

        # 3. Local embeddings + in-memory Chroma (avoids SQLite file locking).
        print(f"[*] Loading local embedding model ({EMBEDDING_MODEL})...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="hk_credit_risk_policies",
            # Keep the "nothing leaves this machine" promise: no Chroma telemetry.
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.retriever = vector_db.as_retriever(search_kwargs={"k": k})
        print("[*] Embeddings generated and stored in-memory.")

        # 4. Local generation LLM.
        print(f"[*] Initializing local LLM via Ollama (model: {model})...")
        self.llm = Ollama(model=model, temperature=0.0)

    def classify_intent(self, query: str) -> str:
        """LLM-based intent router; defaults to 'informational' on ambiguity."""
        verdict = self.llm.invoke(INTENT_PROMPT.format(question=query))
        return "compliance" if "COMPLIANCE" in verdict.upper() else "informational"

    def ask(self, query: str) -> RAGResult:
        # Retrieve ONCE; the same chunks are shown to the user, sent to the
        # generator, and handed to the judge — no display/generation drift.
        docs = self.retriever.invoke(query)
        context = format_docs(docs)
        intent = self.classify_intent(query)
        template = COMPLIANCE_PROMPT if intent == "compliance" else INFORMATIONAL_PROMPT
        answer = self.llm.invoke(template.format(context=context, question=query)).strip()
        return RAGResult(query=query, intent=intent, context=context, answer=answer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_query(rag: CreditRiskRAG, query: str, show_context: bool) -> None:
    print(f'\n[*] Query: "{query}"')
    result = rag.ask(query)
    print(f"[*] Routed intent: {result.intent}")
    if show_context:
        print("\n[*] Retrieved Context Snippets (from local ChromaDB):")
        print(result.context)
    print("\n[*] Generated Assessment:")
    print(f"\n{result.answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="100% local HK banking credit-risk RAG.")
    parser.add_argument("query", nargs="*", help="question to ask (default: built-in demo query)")
    parser.add_argument("-i", "--interactive", action="store_true", help="interactive REPL mode")
    parser.add_argument("-k", type=int, default=5, help="chunks to retrieve (default: 5)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--data-dir", default="data", help="directory of policy PDFs (default: data)")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    args = parser.parse_args()

    print("--- HK Banking RAG Pipeline (Multi-Document / 100% Local) ---")
    rag = CreditRiskRAG(data_dir=args.data_dir, model=args.model, k=args.k)

    if args.interactive:
        print("\nInteractive mode — Ctrl-D or 'exit' to quit.")
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in {"exit", "quit"}:
                break
            run_query(rag, query, args.show_context)
    else:
        run_query(rag, " ".join(args.query) or DEMO_QUERY, args.show_context)


if __name__ == "__main__":
    main()

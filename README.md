# Advanced Banking RAG System (Local Deployment)

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline tailored for banking credit risk policies, **plus an LLM-as-Judge evaluation harness that scores the pipeline's own answers**. The system operates **entirely offline** with zero data leaving your machine, making it suitable for highly confidential financial environments.

## What This Is

This repository is a self-contained solution to a two-part take-home assignment:

| Task | Deliverable | Where it lives |
|---|---|---|
| **1. Simple End-to-End RAG** — ingestion & chunking, embeddings & vector storage, retrieval, LLM generation | A runnable pipeline over 3 synthetic HK banking policy PDFs | `credit_risk_rag.py` (+ `generate_mock_data.py` for the corpus) |
| **2. LLM-as-Judge Assessment Prompt** — a prompt that evaluates the RAG system's output, with explicitly specified judge inputs | The judge prompt **and** a harness that actually executes it against a golden dataset | `banking_revised_judge_prompt.md` + `evaluate.py` + `eval_dataset.json` |

Everything runs locally: HuggingFace embeddings (CPU), in-memory ChromaDB, and Ollama for generation *and* judging. No API keys, no cloud calls.

## Quickstart (one command)

Prerequisite: **[Ollama](https://ollama.com)** installed and running (`ollama serve`). That's it — the script bootstraps everything else, including Python itself, via [uv](https://docs.astral.sh/uv/).

```bash
./run_e2e.sh
```

That single script (macOS/Linux) installs uv if missing, syncs a locked Python environment (`uv sync --frozen` — reproducible to the hash, no system Python or `python3-venv` required), verifies Ollama is up (pulling `llama3.1` if needed), generates the mock policy PDFs, runs the cross-document RAG demo query, and finishes with the full LLM-as-judge evaluation scorecard. Re-running skips completed steps. On Windows, follow the equivalent manual steps in [Setup & Execution](#setup--execution) below.

Short on RAM or CPU? Run with a smaller model: `LOCAL_MODEL_NAME=llama3.2:3b ./run_e2e.sh`.

Expected runtime on first run: a few minutes of downloads (CPU-only PyTorch wheel ~200MB, the MiniLM embedding model ~90MB, and the Ollama model if not yet pulled), then the inference phase. The lockfile pins Linux to CPU-only PyTorch, avoiding ~3GB of unused CUDA libraries that the default PyPI wheel would pull in.

### System requirements & what to expect

| | Default (`llama3.1`, 8B) | Light (`LOCAL_MODEL_NAME=llama3.2:3b`) |
|---|---|---|
| Free RAM during inference | ~6 GB | ~3 GB |
| Disk (Ollama model) | ~4.9 GB | ~2 GB |
| Full e2e on CPU (no GPU) | ~30–40 min | ~15–20 min |
| Full e2e on Apple Silicon / GPU | ~3–5 min | ~2–3 min |

Plus ~2 GB disk for the Python environment (`.venv/`) and ~100 MB of one-time caches. **Output cadence:** setup stages stream instantly; each LLM call then runs silently for 30s–2min on CPU before its result prints — the `[n/5]` progress lines during evaluation are the heartbeat. Don't Ctrl-C during a quiet stretch; it's thinking, not stuck.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama is not reachable` | Install from [ollama.com](https://ollama.com), then `ollama serve` (the desktop app starts it automatically) |
| Model pull fails / out of disk | `ollama pull llama3.2:3b` and re-run with `LOCAL_MODEL_NAME=llama3.2:3b` |
| Inference extremely slow or machine swapping | Use the smaller model (see above) — the 8B default wants ~6 GB free RAM |
| Embedding model download fails (`huggingface.co` blocked) | Opt into a mirror: `HF_ENDPOINT=https://hf-mirror.com ./run_e2e.sh` |
| `uv: command not found` after the script installed it | Open a new shell, or `export PATH="$HOME/.local/bin:$PATH"` |
| Windows | Use WSL for `run_e2e.sh`, or follow the manual `pip` steps in [Setup & Execution](#setup--execution) |

---

## Repository Structure & File Overview

*   **`run_e2e.sh`**
    One-command end-to-end runner (macOS/Linux): environment setup → Ollama preflight → data generation → RAG demo → judge evaluation. Idempotent; supports `LOCAL_MODEL_NAME=<model>` and `--skip-eval`.
*   **`generate_mock_data.py`**
    A utility script that programmatically builds realistic, multi-page PDF documents. It uses the `fpdf2` library to format standard text into actual PDFs to simulate the messy reality of parsing corporate documents.
*   **`credit_risk_rag.py`**
    The core application. It loads the PDFs, splits them into semantic chunks while tracking page numbers and filenames, embeds them using a local HuggingFace model, stores them in ChromaDB, **routes each query by intent (compliance check vs. informational)**, and queries a local Ollama LLM to answer complex risk questions with strict citations. Importable as a library (`CreditRiskRAG.ask()`) or runnable as a CLI.
*   **`banking_revised_judge_prompt.md`**
    The LLM-as-Judge system prompt. It explicitly declares which inputs the judge evaluates against (`original_query`, `retrieved_context`, optional `ground_truth`, `generated_answer`), scores four metrics (faithfulness, helpfulness, **citation accuracy**, operational efficiency/BLUF), raises banking-specific critical risk flags (PII leakage, fair-lending, AML tipping-off), and emits a deterministic PASS/FAIL verdict as JSON.
*   **`evaluate.py`**
    The evaluation harness that actually *executes* the judge prompt: it runs every case in the golden dataset through the RAG pipeline, sends the (query, retrieved context, ground truth, answer) tuple to a judge LLM, parses the JSON verdicts, and prints a scorecard. Exits non-zero on any FAIL, so it can gate CI.
*   **`eval_dataset.json`**
    A small golden dataset of 5 cases: cross-document violation detection, mathematical exception handling (LTV reduction), threshold extraction (UBO), stress-testing rules, and an intentionally **unanswerable question** to verify the system refuses rather than hallucinates.
*   **`pyproject.toml` + `uv.lock` + `.python-version`**
    The canonical dependency definition. `uv sync --frozen` reproduces the exact environment (hash-pinned lockfile, Python 3.12 auto-provisioned by uv, CPU-only PyTorch on Linux). No API-based SDKs — the stack is fully local.
*   **`requirements.txt`**
    Fallback for plain `pip` users (e.g., on Windows without uv); mirrors the pins in `pyproject.toml`.
*   **`data/` (Folder)**
    Where the generated synthetic PDF policy documents live. Every PDF in this folder is ingested at startup.

---

## Local Architecture & Advanced Features

1. **Local Embeddings:** Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` to vectorize PDF chunks natively on the CPU (one-time ~90MB model download, cached afterwards). If `huggingface.co` is blocked on your network, opt into a mirror: `HF_ENDPOINT=https://hf-mirror.com ./run_e2e.sh`.
2. **Local Vector Database:** `ChromaDB` in *ephemeral (in-memory)* mode, which sidesteps SQLite file-locking and sandbox crashes on macOS. Retrieves the top `k=5` chunks (tunable via `-k`) for cross-document queries.
3. **Local LLM Generation:** `Ollama` runs the generator (default `llama3.1`, override with `--model` or `LOCAL_MODEL_NAME`) directly on local hardware. The pipeline **fails fast with actionable errors** if Ollama is unreachable or the model isn't pulled — before spending time on embedding.
4. **LLM-Based Intent Routing:** Each query is first classified as `compliance` (a concrete deal/client scenario to adjudicate) or `informational` (a question about what the policy says). Compliance queries get the hardened "list every violation" prompt; informational queries get a direct-answer prompt — eliminating the awkward "Policy Violation: None found" boilerplate on simple lookups.
5. **Single-Pass Retrieval:** Chunks are retrieved once per query; the same set is displayed to the user, fed to the generator, and handed to the judge — so what you audit is exactly what the model saw.
6. **Multi-Document Citations:** The prompt forces strict `[Document Name - Page N]` citations sourced from chunk metadata, and the judge independently verifies every citation exists in the retrieved context and supports its claim.

---

## The `data/` Folder & How Data is Mocked

The data is procedurally generated via `generate_mock_data.py` to prove that the pipeline handles native PDF ingestion rather than clean, pre-formatted Markdown.

The `data/` folder contains three synthetic policies with overlapping risk criteria:

1.  **`HK_Commercial_Real_Estate_Policy_2024.pdf`** — LTV and DSCR limits by property type, the >HKD 500M LTV reduction rule, HKMA-style stress testing (+200bps, stressed DSCR ≥ 1.0x), and non-recourse lending restrictions.
2.  **`SME_Unsecured_Lending_Guidelines.pdf`** — hard constraints for unsecured business loans: max HKD 5,000,000, max 60-month tenor, restricted industries, mandatory personal guarantees.
3.  **`AML_KYC_Compliance_Manual.pdf`** — UBO verification thresholds (25% standard risk, 10% high risk), PEP Enhanced Due Diligence with FCC/CRO sign-off, and SAR triggers.

---

## Setup & Execution

> **Shortcut (macOS/Linux):** `./run_e2e.sh` performs all of the steps below in one go. The manual steps are listed for Windows users and for anyone who wants to run the stages individually.

### 1. Prerequisites
*   **Ollama** installed (download from [ollama.com](https://ollama.com/))
*   **uv** ([install docs](https://docs.astral.sh/uv/getting-started/installation/)) — provisions Python 3.12 and the locked environment automatically. *Alternative:* any Python 3.10+ with plain `pip` and `requirements.txt`.

**macOS notes:** `brew install ollama uv` covers both prerequisites (or use the ollama.com app, which auto-starts the server — no `ollama serve` needed). Apple Silicon runs the models on the Metal GPU, so expect the fast column in the timing table. Intel Macs are supported too — the lockfile pins `torch 2.2.2` there (the last release with Intel-macOS wheels), which uv selects automatically.

### 2. Prepare the Local LLM (Ollama)
```bash
ollama pull llama3.1        # or a smaller model, e.g.: ollama pull llama3.2:3b
```
*(Make sure the Ollama service is running afterwards — `ollama serve` if it isn't already.)*

### 3. Set Up the Python Environment

**With uv (any OS):**
```bash
uv sync --frozen
```

**Plain pip fallback (e.g., Windows):**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
*(On Linux with pip, first run `pip install torch --index-url https://download.pytorch.org/whl/cpu` to avoid ~3GB of CUDA downloads.)*

### 4. Run the Pipeline
```bash
# Generate the PDF files in the data/ directory
uv run generate_mock_data.py

# Run the built-in cross-document demo query
uv run credit_risk_rag.py

# Ask your own question (with retrieved chunks shown)
uv run credit_risk_rag.py --show-context "What is the max LTV for a Grade A office in Central valued at HKD 600 million?"

# Interactive REPL over the policy corpus
uv run credit_risk_rag.py --interactive
```
*(pip users: replace `uv run` with `python` inside the activated venv.)*

### 5. Evaluate with LLM-as-Judge
```bash
# Judge every golden case (judge model defaults to the generator model)
uv run evaluate.py

# Better practice: judge with a stronger/different model to avoid self-preference bias
uv run evaluate.py --judge-model qwen2.5

# Debug a single case
uv run evaluate.py --case ltv_math_exception
```
The harness prints a scorecard (0–2 per metric, per case), average scores, critical-flag counts, and a PASS/FAIL verdict per case, followed by the judge's full JSON rationales. It exits non-zero if any case fails, so it can gate a CI pipeline.

**Judge inputs (explicitly specified in the prompt):** the judge compares the `generated_answer` against the `original_query`, the `retrieved_context` snippets, and the `ground_truth` when available — and is instructed that its own world knowledge must never override the retrieved context.

**Metrics:** ① Faithfulness/hallucination ② Helpfulness/ground-truth alignment (with a `null` path for correct refusals) ③ Citation accuracy (fabricated citations are an automatic fail) ④ Operational efficiency/BLUF. Plus circuit-breaker flags for PII leakage, fair-lending violations, and AML tipping-off.

**Trust-but-verify:** each critical flag must carry a verbatim quote from the generated answer as evidence. The harness checks that quote programmatically — a flag whose "evidence" doesn't actually appear in the answer is discarded (marked `*` in the scorecard), and the PASS/FAIL verdict is recomputed from the prompt's deterministic rule rather than taken from the judge. This was added after observing a small judge model (llama3.2:3b) assert an AML tipping-off flag while "quoting" the rubric instead of the answer.

---

## Example Queries Supported

*   **Cross-document policy violations:** *"A client wants an unsecured SME loan of HKD 8 million for 72 months. One of their directors is a PEP. What are the policy violations?"* → routed to the **compliance** prompt; synthesizes SME limits + AML PEP rules with citations.
*   **Mathematical reasoning & exceptions:** *"What is the maximum LTV for a Grade A office in Central valued at HKD 600 million?"* → routed to the **informational** prompt; applies the >HKD 500M reduction note (50% − 10% = 40%) without appending a spurious violations section.
*   **Threshold extraction:** *"Three owners hold 30%, 20%, and 50%. Which must be verified?"* → applies the 25% UBO threshold to each stake.
*   **Correct refusal:** *"What is the maximum credit card limit for private banking clients?"* → not in any policy; the grounded prompt forces *"The policy does not specify this."*

---

## Production Roadmap

This MVP demonstrates 100% local, offline multi-document reasoning with a built-in evaluation loop. Remaining architectural improvements before a production banking deployment:

1. ~~**Intent Routing (Dynamic Prompting)**~~ — **Implemented.** An LLM-based classifier routes queries to a `Compliance` or `Informational` prompt, so informational lookups no longer receive forced violation analyses.
2. **Self-Querying Retrieval (Metadata Filtering)** — semantic search currently spans the whole corpus; a query analyzer should translate natural language into metadata filters (e.g., `filter={"source": "HK_Commercial_Real_Estate_Policy_2024.pdf"}`) to prevent cross-policy bleed.
3. **Agentic RAG for Mathematical Constraints** — banking policy math (LTV reductions, DSCR coverage) should be offloaded to a calculator/Python tool rather than trusted to LLM arithmetic. Note the judge already independently re-verifies arithmetic, so failures here are *detected* even before they are *prevented*.
4. **Decoupled Vector Database Architecture** — ingestion currently rebuilds the in-memory Chroma index on every run. Production should deploy a centralized vector DB (Milvus/Qdrant) with an asynchronous pipeline that re-embeds only changed documents via content hashing.
5. **Larger Golden Dataset & Judge Calibration** — extend `eval_dataset.json` with adversarial cases (near-miss numbers, conflicting policies) and periodically spot-check judge verdicts against human review; use a stronger judge model than the generator to avoid self-preference bias.

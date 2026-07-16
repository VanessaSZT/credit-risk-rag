# Advanced Banking RAG System (Local Deployment)

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline tailored for banking credit risk policies. This system operates **entirely offline** with zero data leaving your machine, making it perfect for highly confidential financial environments.

---

## Repository Structure & File Overview

*   **`generate_mock_data.py`**
    A utility script that programmatically builds realistic, multi-page PDF documents. It uses the `fpdf2` library to format standard text into actual PDFs to simulate the messy reality of parsing corporate documents.
*   **`credit_risk_rag.py`**
    The core application. It loads the PDFs, splits them into semantic chunks while tracking page numbers and filenames, embeds them using a local HuggingFace model, saves them to ChromaDB, and queries a local Ollama LLM to answer complex risk questions.
*   **`requirements.txt`**
    The list of Python dependencies required to run the pipeline (e.g., `langchain`, `chromadb`, `sentence-transformers`, `fpdf2`, `pypdf`).
*   **`data/` (Folder)**
    The directory where the generated synthetic PDF policy documents are stored. The RAG pipeline continuously watches and ingests all PDFs in this folder.

---

## Local Architecture & Advanced Features

1. **Local Embeddings (with Mirror Fallback):** Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` to vectorize PDF chunks natively on the CPU. The script automatically routes traffic through `hf-mirror.com` to bypass corporate firewalls or regional timeouts.
2. **Local Vector Database:** Uses `ChromaDB` running in *ephemeral (in-memory)* mode. This completely solves SQLite file-locking and sandbox crashes on macOS, providing a blazingly fast, stable database for local development. Retrieves the top `k=5` chunks to ensure sufficient context for cross-document queries.
3. **Local LLM Generation:** Uses `Ollama` to run a conversational LLM (e.g., Llama 3.1) directly on local hardware.
4. **Multi-Document Citations:** Reads all PDFs in a directory and strictly cites the *Document Name* and *Page Number* in the final assessment. A hardened system prompt forces the LLM to synthesize rules across multiple files without conflating them.

---

## The `data/` Folder & How Data is Mocked

The data is procedurally generated via `generate_mock_data.py` to prove that the pipeline can handle native PDF ingestion rather than clean, pre-formatted Markdown. 

The `data/` folder contains three synthetic policies containing overlapping risk criteria:

1.  **`HK_Commercial_Real_Estate_Policy_2024.pdf`**
    Contains limits for Loan-to-Value (LTV) and Debt Service Coverage Ratios (DSCR) based on property type (e.g., Grade A Office vs. Retail). Includes rules on HKMA stress testing and strict limits on non-recourse lending.
2.  **`SME_Unsecured_Lending_Guidelines.pdf`**
    Dictates terms for unsecured business loans. Defines hard constraints: Max loan of HKD 5,000,000, Max tenor of 60 months, strict prohibition on restricted industries, and mandatory personal guarantees.
3.  **`AML_KYC_Compliance_Manual.pdf`**
    Outlines Financial Crime Compliance (FCC) rules. Defines Ultimate Beneficial Owner (UBO) verification thresholds (25% for standard risk, 10% for high risk) and outlines extreme constraints for Politically Exposed Persons (PEPs).

---

## Example RAG Processes & Supported Queries

Because the vector database indexes all three documents, the RAG system supports complex, cross-document reasoning. You can modify the `query` variable in `credit_risk_rag.py` to ask various questions.

### Example 1: Cross-Document Policy Violations
**User Query:** 
> "A client wants an unsecured SME loan of HKD 8 million for 72 months. One of their directors is a Politically Exposed Person (PEP). What are the policy violations here according to the SME and AML guidelines?"

**Retrieved Context (Top 5 Chunks):**
*   `[Source Metadata: SME_Unsecured_Lending_Guidelines.pdf - Page 1]` (Loan Limits)
*   `[Source Metadata: SME_Unsecured_Lending_Guidelines.pdf - Page 1]` (Tenor & Repayment)
*   `[Source Metadata: HK_Commercial_Real_Estate_Policy_2024.pdf - Page 1]` (Guarantors)
*   `[Source Metadata: HK_Commercial_Real_Estate_Policy_2024.pdf - Page 1]` (LTV Matrix)
*   `[Source Metadata: AML_KYC_Compliance_Manual.pdf - Page 1]` (UBO & Scope)

**LLM Output (from local Ollama - llama3.1):**
> Based on the provided context, I have identified the following policy violations:
> 
> 1. **Loan Limit Exceeded**: The client's requested loan amount of HKD 8 million exceeds the maximum allowed limit of HKD 5,000,000 as stated in [SME_Unsecured_Lending_Guidelines.pdf - Page 1].
> 2. **Tenor Exceeded**: The client's requested tenor of 72 months exceeds the maximum allowed tenor of 60 months (5 years) as stated in [SME_Unsecured_Lending_Guidelines.pdf - Page 1].
> 3. **PEP Requirement Not Met**: As one of the directors is a Politically Exposed Person (PEP), we must identify and verify the identity of any individual holding a 25% or greater ownership stake, according to [AML_KYC_Compliance_Manual.pdf - Page 1]. However, this requirement may not be applicable for SME unsecured facilities as there is no specific mention of PEP requirements in [SME_Unsecured_Lending_Guidelines.pdf - Page 1].
> 4. **Guarantor Requirement Not Met**: Although the policy requires a personal guarantee from the primary business owners, it does not specify any additional requirements for PEPs or high-risk clients.
> 
> **Policy Violations:**
> * Loan Limit Exceeded (SME_Unsecured_Lending_Guidelines.pdf - Page 1)
> * Tenor Exceeded (SME_Unsecured_Lending_Guidelines.pdf - Page 1)
> * PEP Requirement Not Met (AML_KYC_Compliance_Manual.pdf - Page 1)

### Example 2: Mathematical Reasoning & Exceptions
**User Query:** 
> "What is the maximum LTV for a Grade A office in Central valued at HKD 600 million?"

**LLM Output (from local Ollama - llama3.1):**
> According to [HK_Commercial_Real_Estate_Policy_2024.pdf - Page 1], the maximum LTV for a Grade A Office (Central/Admiralty) is 50%.
> 
> However, since the property value exceeds HKD 500 million, we must apply the note on page 1 of the same document: "Note: For properties valued over HKD 500 million, the Max LTV is automatically reduced by 10% from the baseline."
> 
> Therefore, the maximum LTV for a Grade A office in Central valued at HKD 600 million would be 50% - 10% = 40%.
> 
> Policy Violation:
> - The policy does not specify this.

### Example 3: KYC/UBO Extraction (Current Script Output)
**User Query:** 
> "We are onboarding a standard risk corporate client. They have three owners holding 30%, 20%, and 50% stakes respectively. Which individuals must have their identities verified?"

**LLM Output (from local Ollama - llama3.1):**
> According to [AML_KYC_Compliance_Manual.pdf - Page 1], for Standard Risk Corporate Clients, we must identify and verify the identity of any individual holding a 25% or greater ownership stake.
>
> Since two owners hold more than 25% (30% and 50%), their identities must be verified. The owner with a 20% stake does not meet this threshold.
>
> Therefore, the individuals who must have their identities verified are:
> * Owner with 30% stake
> * Owner with 50% stake
>
> Policy Violation: None found in this context.

---

## Setup & Execution

### 1. Prerequisites
*   **Python 3.9+** installed on your machine.
*   **Ollama** installed (Download from [ollama.com](https://ollama.com/)).

### 2. Prepare the Local LLM (Ollama)
Before running the Python script, pull the language model. Open a terminal/command prompt and run:
```bash
ollama run llama3.1
```
*(Once it finishes downloading and gives you a chat prompt, you can type `/bye` to exit. The Ollama service will remain running in the background).* 

### 3. Setup the Python Environment

**For Windows:**
Open Command Prompt or PowerShell in the `Assignment1` folder and run:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**For macOS / Linux:**
Open Terminal in the `Assignment1` folder and run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Pipeline
With your virtual environment activated, generate the mock data and run the pipeline!

```bash
# Generate the PDF files in the data/ directory
python generate_mock_data.py

# Run the cross-document RAG query
python credit_risk_rag.py
```

---

## Evaluation & Production Roadmap

While this system successfully demonstrates 100% local, offline multi-document reasoning, it is an MVP. Before deploying to a production banking environment, the following architectural improvements must be implemented:

### 1. Intent Routing (Dynamic Prompting)
Notice in Example 2 and Example 3 how the LLM awkwardly added: *"Policy Violation: The policy does not specify this"* or *"Policy Violation: None found in this context"* at the end? Because the system prompt is statically hardened to search for "Policy Violations," if a user asks a purely informational question (e.g., "What is the LTV limit?" or "Who needs to be verified?"), the LLM will blindly obey the prompt and attempt to list violations that don't exist. 
**Improvement:** Implement an LLM-based Intent Classifier that routes queries to specific prompts (e.g., `Informational Prompt` vs. `Underwriting Compliance Prompt`).

### 2. Self-Querying Retrieval (Metadata Filtering)
The vector search currently relies purely on semantic similarity across the entire database. This means a query about "Commercial Real Estate" might accidentally retrieve chunks from the "SME Lending" policy if the keywords align.
**Improvement:** Introduce a query-analyzer that translates the user's natural language into strict metadata filters (e.g., `filter={"source": "HK_Commercial_Real_Estate_Policy_2024.pdf"}`).

### 3. Agentic RAG for Mathematical Constraints
Banking policies rely heavily on math (e.g., reducing LTV by 10%, calculating DSCR coverage, threshold comparisons). Standard LLMs hallucinate arithmetic.
**Improvement:** Upgrade the LangChain pipeline to an Agent that has access to a `Calculator` or `Python REPL` tool, allowing the LLM to offload policy math to deterministic code.

### 4. Decoupled Vector Database Architecture
Currently, the script ingests PDFs and rebuilds the Chroma database in-memory on every run to avoid SQLite locking issues.
**Improvement:** Decouple ingestion from querying. Deploy a centralized Vector Database (e.g., Milvus or Qdrant via Docker) and build an asynchronous data pipeline that only embeds newly updated policies using document hashing.

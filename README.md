# RAGdb: The Portable Multimodal Knowledge Container


![PyPI](https://img.shields.io/pypi/v/ragdb)
![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)
![Python Versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)

![downloads](https://img.shields.io/pypi/dw/ragdb)
![downloads_total](https://pepy.tech/badge/ragdb)

**RAGdb is a state-of-the-art, serverless, embedded database designed for offline Retrieval-Augmented Generation (RAG).**

Unlike traditional vector databases that require cloud infrastructure, Docker containers, or heavy GPU dependencies, RAGdb consolidates **automated ingestion, multimodal extraction, vector storage, and hybrid retrieval** into a single, portable SQLite-based container (`.ragdb`).

It serves as the "Long-Term Memory" for AI agents, running entirely on local hardware, edge devices, or within application binaries.

---

## 🚀 Key Capabilities

### 1. Unified Ingestion Pipeline
Automatically detects, parses, and normalizes content from a wide array of unstructured data sources into a structured knowledge graph.
*   **Documents:** `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`, `.xlsx`
*   **Media:** Images (OCR), Audio/Video (Metadata tagging)
*   **Code:** `.py`, `.js`, `.html`, `.xml`

### 2. State-of-the-Art Hybrid Search
RAGdb implements a **Hybrid Retrieval Engine** that outperforms simple cosine similarity by combining:
*   **Dense Vector Search:** TF-IDF weighted vectors for semantic relevance.
*   **Sparse Keyword Search:** Exact substring boosting for precise entity matching (e.g., finding specific invoice numbers or names).

### 3. Zero-Infrastructure Architecture
*   **No Vector DB Server** (Replaces Pinecone/Weaviate/Milvus)
*   **No Heavy ML Frameworks** (Zero PyTorch/Transformers dependency by default)
*   **Single-File Portability:** The entire database is a single file that can be emailed, version-controlled, or embedded.

---

## 📦 Installation

RAGdb is modular. Install only what you need.

### Option A: Lightweight Core (< 30MB)
Best for text, documents, code, and structured data.
```bash
pip install ragdb
```

### Option B: Full Multimodal SOTA (~100MB)
Includes **RapidOCR (ONNX)** for high-fidelity text extraction from images and scanned PDFs.
```bash
pip install "ragdb[ocr]"
```

---

## ⚡ Quick Start

### 1. Build a Knowledge Base
RAGdb uses **Incremental Ingestion**. It hashes files and only processes new or modified content, making it efficient for large directories.

```python

import os
from ragdb import RAGdb

# 1. Initialize your single-file database
db = RAGdb("my_knowledge.ragdb")

# 2. Ingest a folder (PDFs, Images, Text, Excel, etc.)
# This is incremental - run it 100 times, it only updates changed files.
print("📂 Ingesting documents...")
db.ingest_folder("./my_documents")

```

### 2. Perform Hybrid Search
```python

# 1. List what's currently in the DB
print("\n📋 Recent Documents:")
docs = db.list_documents(limit=5)
for path, mtype, updated_at in docs:
    print(f"[{updated_at}] {mtype}: {path}")

# 2. Remove a file (Clean up)
# This updates the vector space immediately.
print("\n🗑️ Deleting old report...")
db.delete_file("./uploads/old_report.pdf")


# 3. Search (State-of-the-Art Hybrid Search)
# Finds "invoice" (semantic) and "INV-2024" (exact match) simultaneously.
query = "invoice for marketing services"
results = db.search(query, top_k=3)

print(f"\n🔍 Top results for: '{query}'")
for res in results:
    print(f"--- Score: {res.score:.4f} | Type: {res.media_type} ---")
    print(f"File: {res.path}")
    print(f"Preview: {res.content[:150]}...")
print("-" * 40)
```

---

## 🤖 Integration: RAG with LLMs

RAGdb is designed to be the retrieval backend for Large Language Models (OpenAI, Claude, LlamaCPP).

```python
# Prerequisite: pip install openai

import os
from openai import OpenAI
from ragdb import RAGdb

# 1. Setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = RAGdb("company_wiki.ragdb")

# (Optional) Ensure we have data
# db.ingest_folder("./company_docs")

def ask_bot(user_question):
    print(f"\n🤔 User asks: {user_question}")

    # --- Step A: RETRIEVAL (The RAGdb Part) ---
    # Get the top 3 most relevant chunks from your local files
    results = db.search(user_question, top_k=3)
    
    if not results:
        return "I couldn't find any information about that in your documents."

    # Create a "Context Block" to feed the AI
    context_text = "\n\n".join([
        f"Source ({r.path}):\n{r.content}" 
        for r in results
    ])

    # --- Step B: GENERATION (The OpenAI Part) ---
    system_prompt = (
        "You are a helpful assistant. Answer the user's question "
        "strictly based on the context provided below."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_question}"}
        ]
    )

    return response.choices[0].message.content

# Run it
answer = ask_bot("What are the payment terms for the project?")
print(f"🤖 AI Answer:\n{answer}")
```

---

## 🛠 Technical Architecture

RAGdb operates on a novel **"Ingest-Normalize-Index"** pipeline:

1.  **Detection:** Magic-byte analysis determines file modality.
2.  **Extraction:** 
    *   *Text:* Utf-8 normalization.
    *   *OCR:* ONNX-based runtime (if enabled) for edge-optimized image processing.
    *   *Tables:* Structure preservation for CSV/Excel.
3.  **Vectorization:** Sublinear Term-Frequency scaling with Inverse Document Frequency (IDF) weighting.
4.  **Storage:** ACID-compliant SQLite container with Write-Ahead Logging (WAL) enabled for concurrency.

---

## 🌐 API Server (Optional)

Expose your `.ragdb` file as a microservice using the built-in FastAPI wrapper.

```bash
pip install "ragdb[server]"
uvicorn ragdb.server:create_app --port 8000
```
*   `POST /ingest`
*   `GET /search?q=...`

---

## 📄 License

This project is licensed under the **Apache 2.0 License**. It is free for commercial use, modification, and distribution.

**Disclaimer:** RAGdb stores extracted knowledge representations, not raw file backups. Always maintain backups of your original source files.

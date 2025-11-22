---
title: 'RAGdb: A Zero-Dependency, Embeddable Architecture for Multimodal Retrieval-Augmented Generation'
tags:
  - Python
  - RAG
  - Retrieval-Augmented Generation
  - SQLite
  - Vector Search
  - Edge AI
authors:
  - name: Ahmed Bin Khalid
    orcid: 0000-0002-0616-2604
    affiliation: 1
affiliations:
 - name: Independent Researcher, Chicago , IL
   index: 1
date: 21 November 2025
bibliography: paper.bib
---

# Summary

Retrieval-Augmented Generation (RAG) is a technique used to ground Large Language Models (LLMs) in external, domain-specific data [@lewis2020retrieval]. However, the standard software stack for implementing RAG has become prohibitively complex, typically requiring a distributed microservices architecture involving an ingestion pipeline, a GPU-backed embedding server, and a dedicated vector database (e.g., Pinecone, Milvus).

`RAGdb` is a Python library that consolidates this entire stack into a single, serverless architecture. It implements a novel "Single-File Knowledge Container" paradigm using SQLite, enabling automated multimodal ingestion (PDF, DOCX, Images), ONNX-based text extraction, and hybrid vector retrieval without requiring external system dependencies or cloud infrastructure.

# Statement of Need

As AI applications move toward the edge (laptops, IoT devices, air-gapped systems), the resource overhead of standard RAG stacks becomes a bottleneck. A typical RAG environment requires installing heavy dependencies like `torch` and `transformers`, often exceeding 3GB in disk space. Furthermore, the reliance on cloud-hosted vector databases raises data sovereignty concerns.

`RAGdb` addresses this need by providing a lightweight (<30MB core), embedded alternative. It is designed for researchers and developers who need to integrate RAG capabilities into local applications, CI/CD pipelines, or privacy-constrained environments where data cannot leave the local machine.

# Architecture and Functionality

`RAGdb` operates on a monolithic architecture where metadata, content, vectors, and indexes are stored within a single ACID-compliant SQLite file.

### Hybrid Retrieval Engine
To eliminate the need for heavy Transformer models at query time, `RAGdb` implements a deterministic **Hybrid Scoring Function (HSF)**. It combines sublinear TF-IDF vectorization [@robertson2009probabilistic] with exact substring boosting:

$$ Score(Q, D) = \alpha \cdot \text{Sim}_{cos}(\vec{v}_Q, \vec{v}_D) + \beta \cdot \mathbb{1}_{substr}(Q, D) $$

This ensures that semantic search is augmented by precise entity retrieval (e.g., finding exact invoice IDs), addressing a common failure mode in pure dense vector retrieval.

### Incremental Ingestion
The library implements a state-based hashing algorithm to track file provenance. This reduces the time complexity of re-indexing a corpus from $O(N)$ (total files) to $O(U)$ (updated files), enabling continuous background synchronization with negligible CPU impact.

# Benchmarks

Experimental evaluation on consumer hardware (Intel i7-1165G7) against a standard Docker-based stack (ChromaDB + LangChain) demonstrates significant efficiency gains:

| Metric | Standard Docker Stack | RAGdb | Improvement |
| :--- | :--- | :--- | :--- |
| **Disk Footprint** | > 1.2 GB | **~5 MB** | **99.5% Reduction** |
| **Query Latency** | ~120 ms | **~60 ms** | **2x Faster** |
| **Incremental Update** | 14.59s (Full Re-index) | **0.46s** | **31.6x Faster** |

These results align with the principles of "Green AI" [@schwartz2020green], reducing the computational cost of retrieval operations.

# References

# RAGdb: A Zero-Dependency, Single-File Database for Multimodal Retrieval-Augmented Generation on the Edge

**Author:** Ahmed Bin Khalid  
**Affiliation:** Independent Researcher  
**Email:** ahmed.khalid2108@gmail.com  
**Date:** November 2025  

## Summary

Retrieval-Augmented Generation (RAG) has become a foundational technique for grounding large language models (LLMs) in external knowledge while reducing hallucinations. However, most modern RAG pipelines depend heavily on GPU-based embedding servers, cloud vector databases, microservice orchestration layers, and multi-gigabyte ML frameworks. These systems are impractical for edge devices, offline deployments, low-resource environments, and privacy-restricted domains.

**RAGdb** is a zero-dependency, embeddable Python library that consolidates multimodal ingestion, sparse vectorization, indexing, and retrieval into a **single portable SQLite file**. The system uses ONNX-based OCR, sublinear TF-IDF vectorization, and a hybrid scoring function to eliminate GPU requirements entirely. This enables reproducible, deterministic retrieval on CPU-only systems with minimal overhead.

## Statement of Need

Most existing RAG systems require:

- GPU inference servers  
- Cloud vector databases (Pinecone, Weaviate, Milvus)  
- Heavy ML dependencies (PyTorch, TensorFlow, CUDA)  
- Distributed ingestion pipelines  
- Docker/Kubernetes orchestration  

These requirements limit adoption in:

- Air‑gapped environments  
- Edge and embedded devices  
- Local, offline knowledge assistants  
- Privacy‑sensitive industries (healthcare, finance, government)  
- Lightweight LLM applications and CI/CD pipelines  

RAGdb provides a **single-file, CPU-only RAG backend** that requires **no external services**, **no GPU**, and **no ML frameworks**, enabling deployment anywhere.

## Features

### Single-File Knowledge Container
All metadata, text content, vectors, and inverted indexes are stored inside a single ACID-compliant SQLite database.

### Automated Multimodal Ingestion
Supports:
- PDF, DOCX, HTML, TXT  
- CSV/Excel (flattened with semantic headers)  
- Images → OCR with ONNXRuntime  
- JSON and structured text  

### Incremental Ingestion (O(U))
A hashing-based algorithm reprocesses only modified files, reducing re-indexing from `O(N)` to `O(U)`.

### Sparse Vectorization + Hybrid Retrieval
TF-IDF scoring:

```
tf(t,d) = 1 + ln(f_t,d)
idf(t) = ln(N / (1 + df_t)) + 1
```

Hybrid scoring:

```
Score(Q,D) = α · cos(v_Q, v_D) + β · 1_substr(Q,D)
```

Substring boosting ensures perfect retrieval of entity-like strings.

### Zero Dependencies
- No PyTorch / TensorFlow  
- No CUDA / GPU  
- No Docker / cloud services  
- Install size < 30 MB  

## System Architecture

RAGdb represents its internal state as:

```
K = < M, C, V, I >
```

Where:

- **M:** Metadata (paths, timestamps, SHA‑256 hashes)  
- **C:** Normalized text chunks  
- **V:** Sparse TF‑IDF vectors  
- **I:** Inverted index for lexical lookup  

SQLite uses WAL mode for concurrency.

## Example Usage

```python
from ragdb import RAGDatabase

db = RAGDatabase("knowledge.ragdb")
db.ingest_folder("documents/")

results = db.query("UNIQUE_INVOICE_CODE_XYZ_999")
for r in results:
    print(r.path, r.score)
```

## Comparison with Existing Tools

| Feature | RAGdb | Chroma | Milvus | Weaviate | FAISS |
|--------|-------|--------|--------|----------|--------|
| Zero dependencies | ✔ | ✖ | ✖ | ✖ | ✔ |
| Single-file DB | ✔ | ✖ | ✖ | ✖ | ✖ |
| Multimodal ingestion | ✔ | ✖ | ✖ | ✖ | ✖ |
| Incremental ingestion | ✔ | ✖ | Limited | Limited | ✖ |
| Offline capable | ✔ | ✔ | ✖ | ✖ | ✔ |
| GPU required | ✖ | Optional | Required | Optional | ✖ |

## Performance Summary

Hardware:
- Intel i7‑1165G7  
- 16 GB RAM  
- Windows 11  

Dataset:
- 1000 mixed documents with injected unique entity IDs  

### Incremental Ingestion

| Operation | Time | Throughput |
|----------|------|-------------|
| Cold start | 14.59 s | 68.5 docs/s |
| Incremental | 0.46 s | >2100 docs/s |

### Entity Retrieval

Query: `UNIQUE_INVOICE_CODE_XYZ_999`  
- Baseline: inconsistent  
- RAGdb: `doc_500.txt`, score 1.5753  
- Recall@1 = 100%  

### Resource Footprint

| Metric | Docker RAG Stack | RAGdb |
|--------|------------------|--------|
| Disk usage | >1.2 GB | ~5 MB |
| Query latency | ~120 ms | ~60 ms |
| Setup time | ~10 min | <10 s |

## Community Guidelines

Contributions are welcome via GitHub pull requests.  
Issues may be opened for ingestion support, performance improvements, or new file types.

## Acknowledgements

RAGdb draws on classical information retrieval research, sparse vectorization techniques, and embedded database design principles.

## References

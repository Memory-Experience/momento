# Data Loading

The data loading infrastructure is designed to maximize efficiency by separating the expensive embedding and indexing process from the evaluation loop.

## Design Philosophy

The key insight is that **embedding passages is the most time-consuming step** in RAG evaluation. By caching embedded passages to disk, we enable:

- **Rapid metric iteration**: Modify evaluation metrics and re-run without re-embedding
- **Configuration experimentation**: Test different retrieval parameters instantly
- **Reproducible comparisons**: All configurations use identical vector representations

## DatasetLoader

The `DatasetLoader` class provides the primary interface for ingesting datasets into the vector store with intelligent caching.

### Key Features

**Intelligent Caching**: The loader checks for existing pickle files (`doc_to_memory.pkl`, `memory_to_doc.pkl`) and skips ingestion if valid caches exist. Invalid partial states are automatically cleaned.

**ID Mapping**: Maintains bidirectional mappings between dataset document IDs and internal memory IDs, enabling seamless translation during evaluation.

**Memory Efficiency**: Uses `itertuples()` for DataFrame iteration instead of `iterrows()`, providing significantly better performance for large datasets by returning lightweight namedtuples rather than full Series objects.

**Direct API Integration**: Interfaces directly with the core API components (`VectorStoreService`, `MemoryRequest`) to leverage production-grade ingestion pipelines.

### Storage Format

Cached data includes:

- **Vector database** (Qdrant): Embedded passages with full metadata
- **ID mappings** (pickle): Bidirectional doc_id ↔ memory_id mappings
- **Atomic writes**: Pickles saved only after complete ingestion to prevent corruption

## BM25DatasetLoader

A specialized loader for baseline BM25 evaluation that uses `LuceneVectorStoreRepository` instead of neural embeddings, creating Pyserini-based BM25 indices for traditional retrieval baselines.

## Repository Pattern Benefits

The modular architecture demonstrates the power of the repository pattern: by implementing the `VectorStoreRepository` interface, we can seamlessly swap between:

- **Neural retrieval** (Qdrant with embeddings)
- **Traditional retrieval** (Lucene with BM25)
- **Hybrid approaches** (future work)

All without changing the evaluation logic or dataset loading code.

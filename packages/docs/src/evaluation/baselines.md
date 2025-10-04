# Baseline Systems

Baseline systems provide reference points for evaluating the effectiveness of neural approaches. The evaluation framework includes two key baseline components.

## BM25 Retrieval (Pyserini)

The BM25 baseline leverages **Pyserini**, a Python interface to the Anserini information retrieval toolkit, which provides production-grade BM25 implementations.

### LuceneVectorStoreRepository

This custom repository implementation wraps Pyserini's Lucene indexing and searching, providing BM25 retrieval through the same interface as neural vector stores.

**Key Design Decision**: By implementing the `VectorStoreRepository` interface, the BM25 baseline can be used interchangeably with neural vector stores. This demonstrates the power of the repository pattern—the same evaluation code works for both traditional and neural retrieval.

### BM25 Parameters

The default parameters (`k1=1.5`, `b=0.75`) are standard BM25 settings that work well across diverse collections. The system supports parameter tuning for dataset-specific optimization.

## Memory Reciter Model

The `MemoryReciterModel` provides a minimalist generation baseline that simply returns the highest-scoring retrieved passage as the answer, without any transformation or synthesis.

### Limitations

The memory reciter baseline has known weaknesses:

- **No synthesis**: Cannot combine information from multiple passages
- **No refinement**: Returns raw passage text without formatting or summarization
- **Single source**: Ignores potentially relevant information from other retrieved passages

Despite these limitations, it provides a useful reference point for understanding the contribution of neural generation to overall performance.

## Why Both Baselines Matter

Together, BM25 retrieval and memory reciter generation form a **fully traditional baseline** that:

- Requires no neural networks (no embedding model, no LLM)
- Runs extremely fast (important for large-scale evaluation)
- Establishes whether neural approaches improve over traditional IR methods

This baseline is particularly valuable for demonstrating the value proposition of neural RAG systems.

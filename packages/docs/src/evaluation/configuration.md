# Configuration System

The evaluation pipeline uses a Python-based configuration system that leverages async iterators to declaratively specify evaluation runs.

## Design Approach

Rather than external configuration files (YAML, JSON), configurations are defined directly in Python as async generator functions. This approach provides:

- **Type safety**: Full IDE support and type checking
- **Flexibility**: Arbitrary Python logic for conditional configurations
- **Clarity**: All dependencies and settings in one place
- **Composability**: Easy to share common configuration snippets

## Configuration Structure

In `main.py`, the `dataset_configurations()` async iterator yields tuples containing a configuration name, dataset, and evaluation client. Each configuration specifies:

- Embedding model selection
- Dataset loading with caching
- RAG component setup (LLM, vector store, services)
- Evaluation client initialization

## Configuration Variants

### Momento Configuration

The full Momento pipeline uses:

- **Embedding model**: Qwen3-Embedding-0.6B (quantized GGUF format)
- **Generation model**: Qwen3-1.7B-Instruct (quantized GGUF format)

These lightweight models enable local evaluation while maintaining competitive performance. Both models run via llama.cpp with GPU acceleration when available.

### SBert Baseline Configuration

Uses Sentence-BERT (all-MiniLM-L6-v2) embeddings for retrieval with the same Qwen3-1.7B-Instruct model for generation, allowing comparison of embedding approaches while holding generation constant.

### BM25 Baseline Configuration

The baseline uses BM25 retrieval (no embeddings) paired with the Memory Reciter model for generation, providing a traditional IR baseline without neural components.

## Dependency Injection

The configuration system showcases dependency injection at its best:

- **Vector stores** are injected into `VectorStoreService`
- **Embeddings** are injected into vector store repositories
- **LLM models** are injected into RAG services
- **RAG services** are injected into the transcription servicer

This enables easy swapping of components without touching evaluation logic.

## Running Evaluations

The main loop iterates through all configurations, evaluating each independently and saving results to timestamped JSON files in the `runs/` directory.

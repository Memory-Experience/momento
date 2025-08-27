# RAG System Evaluation Framework

A comprehensive evaluation suite for Retrieval-Augmented Generation (RAG) systems, designed to measure performance across multiple dimensions: retrieval quality, generation quality, and system latency.

## Architecture

```
evaluation/
├── orchestrator.py      # Main evaluation orchestrator
├── main.py             # CLI entry point
├── eval_config.yaml    # Configuration file
├── common/             # Core utilities
│   ├── dataset.py      # Abstract dataset interface
│   ├── dataframe_dataset.py  # Pandas implementation
│   ├── metrics.py      # IR evaluation metrics
│   └── io.py          # File I/O utilities
└── pipelines/          # Evaluation workflows
    ├── eval_retrieval.py  # Retrieval evaluation
    ├── eval_generation.py # Generation evaluation
    └── eval_latency.py    # Performance evaluation
```

## Quick Start

### 1. Quick Test (10 queries)
```bash
python main.py --quick-test
```

### 2. Full Evaluation with Config
```bash
python main.py --config eval_config.yaml
```

### 3. Custom Configuration
```bash
python orchestrator.py --dataset msmarco --sample-size 100 --output-dir ./my_results
```

## What Gets Evaluated

### **Retrieval Quality**
- **Precision@K**: How many retrieved documents are relevant
- **Recall@K**: How many relevant documents were retrieved  
- **NDCG@K**: Ranking quality with position awareness
- **MRR**: Mean Reciprocal Rank of first relevant document

### **Generation Quality**
- **Exact Match**: Perfect answer matches
- **F1 Score**: Token-level overlap with ground truth
- **BLEU/ROUGE**: Text similarity metrics

### **System Performance**
- **Retrieval Latency**: P50, P90, P95 percentiles
- **Generation Latency**: P50, P90, P95 percentiles
- **End-to-End Latency**: Total pipeline time

## Configuration

Edit `eval_config.yaml` to customize evaluation:

```yaml
dataset:
  type: "msmarco"           # msmarco | custom
  sample_size: 100          # Number of queries to evaluate

evaluation:
  k_values: [1, 3, 5, 10]   # Top-k for retrieval metrics
  retrieval: true           # Enable retrieval evaluation
  generation: true          # Enable generation evaluation
  latency: true            # Enable latency evaluation

rag_service:
  endpoint: "localhost:50051"  # Your RAG service endpoint
```

## Output

Results are saved in multiple formats:

```
eval_results/
├── evaluation_summary.json    # Complete results
├── retrieval_results.csv      # Detailed retrieval metrics
├── generation_results.csv     # Detailed generation metrics
└── latency_summary.json       # Performance metrics
```

## Integration with Your RAG Service

The evaluator interfaces with your RAG service through the `run_rag_pipeline()` method in `orchestrator.py`. Update this method to call your actual service:

```python
def run_rag_pipeline(self, query: str, k: int = 10) -> Dict[str, Any]:
    # Replace with your RAG service call
    rag_response = your_rag_service.query(query, top_k=k)
    
    return {
        'query': query,
        'retrieved_docs': rag_response.documents,
        'answer': rag_response.answer,
        'timing': rag_response.timing
    }
```

## Dataset Support

### MS MARCO (Built-in)
- Automatically downloads and processes MS MARCO passage dataset
- Standard IR benchmark with queries and relevance judgments

### Custom Datasets
- Support for any dataset with docs/queries/qrels format
- CSV/JSON input formats supported

## Testing

```bash
# Run unit tests
python -m pytest test_ms_dataset.py

# Quick validation
python main.py --quick-test --sample-size 5
```

## Requirements

- Python 3.10+
- pandas, numpy
- ir_datasets (for MS MARCO)
- pyyaml (for configuration)
- Your RAG service running

## Use Cases

1. **Benchmark Comparison**: Compare different retrieval models
2. **A/B Testing**: Evaluate system changes
3. **Performance Monitoring**: Track system performance over time
4. **Research**: Academic evaluation following standard metrics

## Contributing

The framework is designed to be extensible:

- Add new datasets by implementing the `Dataset` interface
- Add new metrics in `common/metrics.py`
- Add new evaluation pipelines in `pipelines/`

## Example Output

```
RAG SYSTEM EVALUATION SUMMARY
============================================================

Dataset: 8,841,823 docs, 100 queries

RETRIEVAL METRICS:
----------------------------------------
precision@5         : 0.8200
recall@5           : 0.4100
ndcg@5             : 0.7650
mrr                : 0.8100

GENERATION METRICS:
----------------------------------------
exact_match        : 0.2400
f1_score          : 0.6800

LATENCY METRICS (ms):
----------------------------------------
retrieval_p50      : 45.20
retrieval_p95      : 125.80
generation_p50     : 890.40
generation_p95     : 2140.60
total_p50          : 945.20
```

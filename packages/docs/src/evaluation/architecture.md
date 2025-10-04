# Evaluation Pipeline Architecture

The evaluation package provides a comprehensive framework for assessing the performance of the Momento RAG system. It implements standard information retrieval and generation metrics, supports multiple datasets, and enables fair comparison between different retrieval and generation strategies.

## Overview

The evaluation pipeline is designed with modularity and efficiency in mind, allowing researchers to:

- **Separate data loading from evaluation**: Passages are embedded once and cached, enabling rapid iteration on metrics and configurations
- **Compare multiple approaches**: Baseline systems (BM25) can be directly compared against neural retrieval methods
- **Evaluate across diverse domains**: Support for general QA (MS MARCO), temporal reasoning (TimelineQA), and lifelog retrieval (OpenLifelogQA)
- **Leverage modular architecture**: The dependency injection and repository pattern enable seamless swapping of vector stores and models

## Architecture Components

### [Data Loading](data_loading.md)

Handles efficient ingestion and caching of datasets into vector stores, with support for both neural embeddings and traditional BM25 indexing.

### [Configuration System](configuration.md)

Python-based configuration through async iterators, enabling declarative specification of evaluation runs with different models and settings.

### [Baseline Systems](baselines.md)

Reference implementations including BM25 retrieval and simple generation baselines for comparative evaluation.

### [Datasets](datasets.md)

Adapters for multiple benchmark datasets spanning different domains and difficulty levels, from general QA to domain-specific lifelog retrieval.

### [Metrics](metrics.md)

Comprehensive implementations of retrieval metrics (Precision@K, Recall@K, MRR, NDCG, MAP, AQWV) and generation metrics (F1, ROUGE-L, faithfulness, semantic similarity).

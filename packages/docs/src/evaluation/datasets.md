# Datasets

The evaluation framework supports three benchmark datasets, each targeting different aspects of RAG system performance.

## MS MARCO

**MS MARCO** (Microsoft MAchine Reading COmprehension) is a large-scale question answering dataset derived from real Bing search queries, released by Microsoft Research.

**Citation**: Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., Rosenberg, M., Song, X., Stoica, A., Tiwary, S., & Wang, T. (2016). MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/)

### Dataset Characteristics

- **Domain**: General web search and factoid QA
- **Size**: We evaluate on a subset due to computational constraints (~1000 queries)
- **Structure**: Each query has gold answers and relevant passage IDs
- **Difficulty**: Varies from simple factoid lookup to multi-hop reasoning

### Adapter Implementation

The `MSMarcoDataset` adapter handles:

- Loading from `ir_datasets` library
- Converting to common DataFrame format
- Filtering to needed documents and queries
- Maintaining relevance judgments (qrels)

## TimelineQA

**TimelineQA** is a synthetic benchmark dataset designed to evaluate question answering over timelines and temporal reasoning in lifelog-style data.

**Citation**: Tan, W., Dwivedi-Yu, J., Li, Y., Mathias, L., Saeidi, M., Yan, J. N., & Halevy, A. (2023). TimelineQA: A Benchmark for Question Answering over Timelines. In _Findings of the Association for Computational Linguistics: ACL 2023_ (pp. 77–91). [https://aclanthology.org/2023.findings-acl.6/](https://aclanthology.org/2023.findings-acl.6/)

### Dataset Characteristics

- **Domain**: Personal timeline events (education, relationships, travel, etc.)
- **Temporal focus**: All events are timestamped, requiring date-aware retrieval
- **Generation**: Synthetically generated using template-based persona creation
- **Categories**: Sparse, medium, and dense (varying event density)

### Our Integration

Rather than using TimelineQA's command-line interface, we integrated the dataset generation logic directly into our evaluation framework. This allows programmatic control over dataset creation and seamless integration with our pipeline.

### Passage Construction

**Event descriptions as passages**: We use the event descriptions themselves (e.g., "graduated from university", "traveled to Paris") as retrievable passages. Each event becomes a single document in the vector store.

**Date enhancement**: We prepend timestamps to each passage and augment questions with temporal information, ensuring that retrieval systems must handle temporal context effectively.

### BM25 Advantage

⚠️ **Important caveat**: Because dates appear in both queries and passages, BM25 performs surprisingly well on TimelineQA. Date tokens provide strong lexical matching signals that may not reflect real-world lifelog retrieval challenges. This makes TimelineQA **less representative** of the target domain than initially hoped.

## OpenLifelogQA

**OpenLifelogQA** is a benchmark dataset for open-ended multi-modal lifelog question answering, designed to evaluate retrieval over real lifelogging data, created by researchers at Dublin City University.

**Citation**: Tran, Q., Nguyen, B., Jones, G. J. F., & Gurrin, C. (2025). OpenLifelogQA: An Open-Ended Multi-Modal Lifelog Question-Answering Dataset. _arXiv preprint arXiv:2508.03583_. [https://arxiv.org/abs/2508.03583](https://arxiv.org/abs/2508.03583)

### Dataset Characteristics

- **Domain**: Real lifelog images and their textual descriptions
- **Multimodal**: Links questions to image IDs (we use text descriptions)
- **Realistic**: Actual questions people ask about their captured life moments
- **Splits**: Train, validation, and test sets

### Why This Dataset Matters

OpenLifelogQA is the **most representative dataset** for our domain because:

- **Real-world queries**: Questions come from actual lifelog usage
- **Natural language**: Descriptions are human-written, not templated
- **Diverse complexity**: From simple ("What did I eat?") to complex ("Who was at the meeting last week?")
- **No date bias**: Unlike TimelineQA, doesn't artificially favor keyword matching

### Data Availability

⚠️ **Licensing note**: We cannot redistribute OpenLifelogQA as it requires agreement with the dataset creators. Researchers wishing to reproduce our results should request access through the official channels.

The dataset consists of:

- `event_description.csv`: Image IDs with textual descriptions
- `{split}_data.csv`: Questions, answers, and relevant image IDs

## Dataset Comparison

| Dataset       | Domain            | Scale (our eval) | Temporal | Representative  |
| ------------- | ----------------- | ---------------- | -------- | --------------- |
| MS MARCO      | General QA        | ~1000 queries    | ❌       | Medium          |
| TimelineQA    | Synthetic lifelog | Variable         | ✅       | Low (BM25 bias) |
| OpenLifelogQA | Real lifelog      | ~1000 queries    | Implicit | **High**        |

## Common Interface

All datasets implement the `DataFrameDataset` interface, providing:

- **Consistent schema**: `docs` (id, content), `queries` (id, text, answer), `qrels` (query_id, doc_id, relevance)
- **Pandas DataFrames**: Enable efficient filtering and manipulation
- **In-memory storage**: All data is loaded into DataFrames for fast access during evaluation

This common interface allows the evaluation pipeline to work seamlessly across all datasets without dataset-specific code.

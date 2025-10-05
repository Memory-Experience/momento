# Evaluation Metrics

The evaluation framework implements a comprehensive suite of metrics for assessing both retrieval and generation quality.

## Retrieval Metrics

Retrieval metrics evaluate how well the system finds relevant documents for a given query.

### Precision and Recall

**Precision@K** measures the fraction of top-K retrieved documents that are relevant:

$$
\text{Precision@K} = \frac{|\text{relevant} \cap \text{retrieved}_{1:K}|}{K}
$$

**Recall@K** measures the fraction of all relevant documents found in the top-K:

$$
\text{Recall@K} = \frac{|\text{relevant} \cap \text{retrieved}_{1:K}|}{|\text{relevant}|}
$$

We report both metrics at K = 1, 3, 5, 10, and 20 to understand performance across different retrieval depths.

**Why these matter**: Precision indicates answer density while recall indicates coverage. The precision-recall trade-off is fundamental to retrieval system design.

### Mean Reciprocal Rank (MRR)

MRR measures the average of reciprocal ranks across all queries:

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank of first relevant document for } q}
$$

Where |Q| is the total number of queries. For a single query, the reciprocal rank is 1/rank of the first relevant document (or 0 if no relevant document is found).


**Why this matters**: For question answering, finding at least one good answer quickly is often sufficient. MRR@K variants show this at different cutoffs.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG is a ranking quality metric that accounts for both relevance and position:

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{relevance}_i}{\log_2(i + 1)}
$$

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

Where IDCG is the ideal DCG if documents were perfectly ranked, and relevance_i is 1 for relevant documents and 0 for non-relevant ones.

**Implementation note**: This differs from the graded relevance variant using $\frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$, which requires multi-level relevance judgments.

**Why this matters**: NDCG rewards systems that rank highly-relevant documents higher, making it more nuanced than binary metrics. It's the standard metric for ranking evaluation.

### Mean Average Precision (MAP)

MAP averages the precision at each relevant document position:

$$
\text{AP} = \frac{1}{|\text{relevant}|} \sum_{k=1}^{N} \text{Precision@k} \cdot \text{rel}(k)
$$

Where rel(k) is 1 if document at rank k is relevant, 0 otherwise.

**Why this matters**: MAP provides a single-number summary of precision-recall performance across all recall levels.

### Average Query Weighted Value (AQWV)

AQWV is a cost-sensitive metric from low-resource information retrieval:

$$
\text{AQWV} = 1 - P_{\text{miss}} - \beta \cdot P_{\text{FA}}
$$

Where:

- $P_{\text{miss}}$ = probability of missing relevant documents
- $P_{\text{FA}}$ = probability of false alarms
- $\beta$ = cost ratio (default: 40.0)



**Non-standard implementation**: Our AQWV implementation **does not use a detection threshold**, which is non-standard. We take this approach because:
1. RAG systems typically return a fixed number of documents rather than making binary detection decisions
2. It simplifies integration with existing retrieval pipelines that don't naturally produce confidence scores
3. It focuses evaluation on ranking quality rather than threshold tuning

In our retrieval setting:
- $P_{\text{miss}}$ = fraction of relevant documents not retrieved in top-K
- $P_{\text{FA}}$ = fraction of retrieved documents that are non-relevant

This treats all retrieved documents as positive detections and all non-retrieved documents as negative detections.

**Why we include this**: AQWV originates from the MATERIAL program for low-resource cross-language information retrieval. While uncommon in standard RAG evaluation, we include it to provide a more complete picture of system performance, particularly regarding the trade-off between finding relevant documents (minimizing misses) and avoiding irrelevant ones (minimizing false alarms). This metric was specifically requested to align with research directions in resource-constrained retrieval scenarios.

## Generation Metrics

Generation metrics evaluate the quality of the synthesized answer.

### Exact Match (EM)

Binary indicator of whether the generated answer exactly matches a gold answer (after normalization).

**Why this matters**: While strict, EM is the clearest signal of correctness for factoid questions.

### Token-level F1

F1 score computed over word overlap between generated and gold answers:

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Why this matters**: More lenient than EM, capturing partial correctness and lexical similarity.

### ROUGE-L F1

F1 based on longest common subsequence (LCS):

$$
\text{ROUGE-L} = \frac{2 \cdot \text{LCS}(gen, gold)}{\text{len}(gen) + \text{len}(gold)}
$$

**Why this matters**: Rewards fluent, order-preserving overlap rather than just bag-of-words matching.

### Answer Relevance

Token F1 between the answer and the original query, measuring topical alignment.

**Why this matters**: Ensures answers stay on-topic and address the question asked.

### Faithfulness Metrics

These metrics evaluate how well the answer is grounded in retrieved documents:

**Support Coverage**: Fraction of answer content words found in retrieved documents.

**Support Density**: Fraction of all answer tokens present in retrieved documents.

**Hallucination Rate**: `1 - Support Density`, indicating fabricated content.

**Why these matter**: Factual correctness requires grounding in retrieved evidence. These metrics detect when models "hallucinate" information not in the context.

### Semantic Similarity

#### SBERT Similarity

Cosine similarity between SBERT embeddings of generated and gold answers.

**Why this matters**: Captures semantic equivalence beyond lexical overlap (e.g., "car" ≈ "automobile").

#### Cross-Encoder Similarity

Direct relevance scoring using a cross-encoder model (e.g., MS-MARCO MiniLM).

**Why this matters**: Cross-encoders jointly encode both texts, capturing deeper semantic relationships than bi-encoders.

## Implementation Notes

All metrics are implemented in `metrics/retrieval_metrics.py` and `metrics/generation_metrics.py` with:

- **Defensive handling**: Graceful handling of edge cases (empty answers, missing gold answers)
- **Normalization**: Text normalization (lowercasing, punctuation removal, article removal)
- **Efficient computation**: Vectorized operations where possible
- **Clear semantics**: Explicit return types and value ranges

## Metric Aggregation

The `RAGEvaluationClient` computes metrics at two levels:

1. **Per-query**: Individual scores for each query, stored in results JSON
2. **Corpus-level**: Averaged scores across all queries, summarizing overall performance

This dual reporting enables both aggregate comparisons and fine-grained error analysis.

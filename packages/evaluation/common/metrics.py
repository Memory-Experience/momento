import math
from typing import List, Set, Dict

def precision_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    top = ranked_ids[:k]
    hits = sum(1 for x in top if x in gold_ids)
    return hits / max(1, k)

def recall_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids: return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for x in top if x in gold_ids)
    return hits / len(gold_ids)

def reciprocal_rank(ranked_ids: List[str], gold_ids: Set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, 1):
        if doc_id in gold_ids:
            return 1.0 / i
    return 0.0

def dcg_at_k(rels: List[int], k: int) -> float:
    return sum(rel / math.log2(i+2) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    # binary relevance
    rels = [1 if d in gold_ids else 0 for d in ranked_ids]
    dcg = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

# =============================================================================
# GENERATION METRICS (for RAG answer quality) #needs to be checked.
# =============================================================================

def exact_match(predicted: str, gold: str) -> float:
    """Exact string match between predicted and gold answers."""
    return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0

def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 score between predicted and gold answers."""
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    intersection = pred_tokens & gold_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def answer_relevance(answer: str, question: str) -> float:
    """
    Simple keyword-based relevance between answer and question.
    In practice, you'd use embedding similarity or LLM-based scoring.
    """
    answer_words = set(answer.lower().split())
    question_words = set(question.lower().split())
    
    if not answer_words:
        return 0.0
    
    overlap = len(answer_words & question_words)
    return overlap / len(question_words) if question_words else 0.0

def context_precision(retrieved_docs: List[str], gold_answer: str) -> float:
    """
    Measures how much of retrieved context is relevant to the gold answer.
    Simple token overlap implementation.
    """
    if not retrieved_docs:
        return 0.0
    
    gold_tokens = set(gold_answer.lower().split())
    relevant_docs = 0
    
    for doc in retrieved_docs:
        doc_tokens = set(doc.lower().split())
        if doc_tokens & gold_tokens:  # If any overlap with gold answer
            relevant_docs += 1
    
    return relevant_docs / len(retrieved_docs)

def context_recall(retrieved_docs: List[str], gold_answer: str) -> float:
    """
    Measures how much of the gold answer is covered by retrieved context.
    """
    if not gold_answer:
        return 0.0
    
    gold_tokens = set(gold_answer.lower().split())
    all_context_tokens = set()
    
    for doc in retrieved_docs:
        all_context_tokens.update(doc.lower().split())
    
    covered_tokens = gold_tokens & all_context_tokens
    return len(covered_tokens) / len(gold_tokens)

def faithfulness(answer: str, retrieved_docs: List[str]) -> float:
    """
    Simple faithfulness check: is the answer supported by retrieved docs?
    Measures token overlap between answer and context.
    """
    if not answer or not retrieved_docs:
        return 0.0
    
    answer_tokens = set(answer.lower().split())
    context_tokens = set()
    
    for doc in retrieved_docs:
        context_tokens.update(doc.lower().split())
    
    if not answer_tokens:
        return 1.0
    
    supported_tokens = answer_tokens & context_tokens
    return len(supported_tokens) / len(answer_tokens)

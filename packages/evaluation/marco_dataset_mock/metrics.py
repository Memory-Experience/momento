"""
Evaluation Metrics Calculator

Standard information retrieval evaluation metrics implementation.
"""

import math
from typing import List, Dict


class EvaluationMetrics:
    """Standard evaluation metrics for information retrieval systems."""
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Precision@k - fraction of top-k results that are relevant.
        
        Args:
            retrieved_ids: Retrieved document IDs in ranking order
            relevant_ids: Ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        
        return relevant_retrieved / min(k, len(retrieved_k))
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Recall@k - fraction of relevant items found in top-k results.
        
        Args:
            retrieved_ids: Retrieved document IDs in ranking order
            relevant_ids: Ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        
        return relevant_retrieved / len(relevant_ids)
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate NDCG@k - Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved_ids: Retrieved document IDs in ranking order
            relevance_scores: Document ID to relevance score mapping
            k: Number of top results to consider
            
        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
            
        retrieved_k = retrieved_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / math.log2(i + 2)
        
        # Calculate IDCG (perfect ranking)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance):
            idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate MRR - Mean Reciprocal Rank.
        
        Args:
            retrieved_ids: Retrieved document IDs in ranking order
            relevant_ids: Ground truth relevant document IDs
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    @classmethod
    def calculate_all_metrics(cls, retrieved_ids: List[str], relevant_ids: List[str], 
                            relevance_scores: Dict[str, float], k_values: List[int]) -> Dict[str, float]:
        """Calculate all metrics for given retrieval results.
        
        Args:
            retrieved_ids: Retrieved document IDs in ranking order
            relevant_ids: Ground truth relevant document IDs
            relevance_scores: Document ID to relevance score mapping
            k_values: K values to calculate metrics for
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = cls.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f'recall@{k}'] = cls.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f'ndcg@{k}'] = cls.ndcg_at_k(retrieved_ids, relevance_scores, k)
        
        metrics['mrr'] = cls.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        
        return metrics

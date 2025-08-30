"""
Evaluation Metrics Calculator

Standardized metrics for information retrieval evaluation.
Consolidated from various scattered implementations.
"""

import math
from typing import List, Dict


class EvaluationMetrics:
    """Standardized evaluation metrics calculator."""
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate Precision@k - how many of the top-k retrieved items are relevant.
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranking order
            relevant_ids: List of relevant document IDs (ground truth)
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
        """
        Calculate Recall@k - how many of the relevant items are in top-k results.
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranking order
            relevant_ids: List of relevant document IDs (ground truth)
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
        """
        Calculate NDCG@k - Normalized Discounted Cumulative Gain.
        Measures ranking quality with position awareness.
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranking order
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Number of top results to consider
            
        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
            
        retrieved_k = retrieved_ids[:k]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) is undefined
        
        # Calculate IDCG (Ideal DCG - perfect ranking)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance):
            idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate MRR - Mean Reciprocal Rank.
        Measures how quickly users find relevant results.
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranking order
            relevant_ids: List of relevant document IDs (ground truth)
            
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
        """
        Calculate all metrics for given results.
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranking order
            relevant_ids: List of relevant document IDs (ground truth)
            relevance_scores: Dictionary mapping doc_id to relevance score
            k_values: List of k values to calculate metrics for
            
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

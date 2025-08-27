"""
Generic Evaluation Framework for Information Retrieval Systems

This framework provides a standardized way to evaluate any IR system using:
- MS MARCO (default)
- Personal Memory datasets
- Custom datasets

Key Components:
1. DatasetInterface: Abstract interface for any dataset
2. EvaluationEngine: Generic evaluation logic
3. MetricsCalculator: Standardized metrics (P@k, R@k, NDCG@k, MRR)
4. DatasetAdapters: Concrete implementations for different data sources
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import logging
from pathlib import Path

class DatasetInterface(ABC):
    """Abstract interface for evaluation datasets."""
    
    @abstractmethod
    def get_queries(self) -> pd.DataFrame:
        """Return queries DataFrame with columns: ['id', 'text']"""
        pass
    
    @abstractmethod
    def get_documents(self) -> pd.DataFrame:
        """Return documents DataFrame with columns: ['id', 'content']"""
        pass
    
    @abstractmethod
    def get_relevance_judgments(self) -> pd.DataFrame:
        """Return relevance judgments DataFrame with columns: ['query_id', 'doc_id', 'relevance']"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return dataset name for logging/reporting"""
        pass

class RetrievalSystemInterface(ABC):
    """Abstract interface for retrieval systems to be evaluated."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Search for documents given a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of document IDs ranked by relevance
        """
        pass

class EvaluationMetrics:
    """Standardized evaluation metrics calculator."""
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Precision@k"""
        if k == 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        return relevant_retrieved / min(k, len(retrieved_k))
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Recall@k"""
        if not relevant_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        return relevant_retrieved / len(relevant_ids)
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate NDCG@k"""
        import math
        
        if k == 0:
            return 0.0
            
        retrieved_k = retrieved_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) is undefined
        
        # Calculate IDCG (perfect ranking)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance):
            idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate MRR (Mean Reciprocal Rank)"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

class GenericEvaluationEngine:
    """
    Generic evaluation engine that can work with any dataset and retrieval system.
    """
    
    def __init__(self, dataset: DatasetInterface, retrieval_system: RetrievalSystemInterface):
        self.dataset = dataset
        self.retrieval_system = retrieval_system
        self.metrics = EvaluationMetrics()
        
    def evaluate(self, sample_size: int = 100, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            sample_size: Number of queries to evaluate
            k_values: List of k values for metrics calculation
            
        Returns:
            Dictionary with evaluation results
        """
        logging.info(f" Starting evaluation on {self.dataset.get_name()}")
        logging.info(f" Sample size: {sample_size}, K values: {k_values}")
        
        queries_df = self.dataset.get_queries()
        qrels_df = self.dataset.get_relevance_judgments()
        
        # Sample queries
        sample_queries = queries_df.head(sample_size)
        
        results = {
            'dataset': self.dataset.get_name(),
            'sample_size': len(sample_queries),
            'k_values': k_values,
            'metrics': {f'precision@{k}': [] for k in k_values},
            'query_results': []
        }
        
        # Add other metrics
        for k in k_values:
            results['metrics'][f'recall@{k}'] = []
            results['metrics'][f'ndcg@{k}'] = []
        results['metrics']['mrr'] = []
        
        for _, query_row in sample_queries.iterrows():
            query_id = query_row['id']
            query_text = query_row['text']
            
            # Get gold standard for this query
            query_qrels = qrels_df[qrels_df['query_id'] == query_id]
            if query_qrels.empty:
                continue
                
            relevant_docs = query_qrels['doc_id'].tolist()
            relevance_scores = dict(zip(query_qrels['doc_id'], query_qrels['relevance']))
            
            # Run retrieval
            try:
                retrieved_ids = self.retrieval_system.search(query_text, top_k=max(k_values))
            except Exception as e:
                logging.warning(f" Search failed for query {query_id}: {e}")
                continue
            
            # Calculate metrics for this query
            query_metrics = {}
            for k in k_values:
                query_metrics[f'precision@{k}'] = self.metrics.precision_at_k(retrieved_ids, relevant_docs, k)
                query_metrics[f'recall@{k}'] = self.metrics.recall_at_k(retrieved_ids, relevant_docs, k)
                query_metrics[f'ndcg@{k}'] = self.metrics.ndcg_at_k(retrieved_ids, relevance_scores, k)
                
                # Add to results
                results['metrics'][f'precision@{k}'].append(query_metrics[f'precision@{k}'])
                results['metrics'][f'recall@{k}'].append(query_metrics[f'recall@{k}'])
                results['metrics'][f'ndcg@{k}'].append(query_metrics[f'ndcg@{k}'])
            
            mrr = self.metrics.mean_reciprocal_rank(retrieved_ids, relevant_docs)
            query_metrics['mrr'] = mrr
            results['metrics']['mrr'].append(mrr)
            
            # Store individual query result
            results['query_results'].append({
                'query_id': query_id,
                'query_text': query_text,
                'retrieved_count': len(retrieved_ids),
                'relevant_count': len(relevant_docs),
                'metrics': query_metrics
            })
        
        # Calculate averages
        results['avg_metrics'] = {}
        for metric_name, values in results['metrics'].items():
            if values:  # Only if we have values
                results['avg_metrics'][metric_name] = sum(values) / len(values)
            else:
                results['avg_metrics'][metric_name] = 0.0
        
        logging.info(f" Evaluation completed on {len(results['query_results'])} queries")
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a nice format."""
        print(f"\n EVALUATION RESULTS - {results['dataset']}")
        print("=" * 60)
        print(f"Sample Size: {results['sample_size']} queries")
        print(f"K Values: {results['k_values']}")
        print("\n AVERAGE METRICS:")
        
        for metric_name, avg_value in results['avg_metrics'].items():
            print(f"  {metric_name}: {avg_value:.4f}")
        
        print(f"\n QUERY BREAKDOWN:")
        print(f"  Evaluated: {len(results['query_results'])} queries")
        print(f"  Skipped: {results['sample_size'] - len(results['query_results'])} queries")

"""
Basic MS MARCO Top-K Evaluation Class

This class provides the missing link between your existing components:
- MSMarcoDataset (loading data)
- EvaluationMetrics (calculating metrics)
- RetrievalSystem (getting top-k results)

It orchestrates the evaluation process to get top-k evaluation results.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Protocol
import time
from collections import defaultdict

from .marco_dataset import MSMarcoDataset
from .metrics import EvaluationMetrics


class RetrievalSystemInterface(Protocol):
    """Interface for retrieval systems to be evaluated."""
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve top-k document IDs for a query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of document IDs ranked by relevance
        """
        ...


class MarcoTopKEvaluator:
    """
    Basic MS MARCO Top-K evaluation class that connects all existing components.
    
    This class:
    1. Takes a loaded MSMarcoDataset
    2. Takes a retrieval system 
    3. Runs queries through the system to get top-k results
    4. Uses EvaluationMetrics to calculate performance scores
    """
    
    def __init__(self, dataset: MSMarcoDataset, k_values: Optional[List[int]] = None):
        """
        Initialize evaluator with MS MARCO dataset.
        
        Args:
            dataset: Loaded MSMarcoDataset instance
            k_values: List of k values to evaluate (default: [1, 3, 5, 10])
        """
        self.dataset = dataset
        self.k_values = k_values or [1, 3, 5, 10]
        
        # Build lookup structures for efficient evaluation
        self.relevance_lookup = self._build_relevance_lookup()
        
        print(f"Marco Top-K Evaluator initialized:")
        print(f"  Dataset: {dataset.get_name()}")
        print(f"  Queries: {len(dataset.queries)}")
        print(f"  Documents: {len(dataset.docs)}")
        print(f"  Relevance judgments: {len(dataset.qrels)}")
        print(f"  K values: {self.k_values}")
    
    # In your eval.py MarcoTopKEvaluator class, update _build_relevance_lookup:
    def _build_relevance_lookup(self) -> Dict[str, Dict[str, float]]:
        """Build efficient query_id -> {doc_id: relevance_score} lookup."""
        lookup = defaultdict(dict)
        
        for _, row in self.dataset.qrels.iterrows():
            query_id = str(row['query_id'])  # Convert to string
            doc_id = str(row['doc_id'])      # Convert to string
            relevance = float(row['relevance'])
            lookup[query_id][doc_id] = relevance
        
        print(f"Built relevance lookup for {len(lookup)} queries")
        # Debug: show sample
        if lookup:
            sample_query_id = list(lookup.keys())[0]
            sample_docs = list(lookup[sample_query_id].keys())[:3]
            print(f"Sample relevance lookup - Query {sample_query_id}: docs {sample_docs}")
        
        return dict(lookup)
    
    def evaluate_query_top_k(self, query_id: str, query_text: str, 
                            retrieval_system: RetrievalSystemInterface) -> Dict[str, Any]:
        """
        Evaluate top-k results for a single query.
        
        Args:
            query_id: Query identifier
            query_text: Query text to search with
            retrieval_system: System to get top-k results from
            
        Returns:
            Dictionary with evaluation results for this query
        """
        # Step 1: Get top-k results from retrieval system
        max_k = max(self.k_values)
        retrieved_doc_ids = retrieval_system.retrieve(query_text, max_k)
        retrieved_doc_ids = [str(doc_id) for doc_id in retrieved_doc_ids]  # Ensure strings
        
        # Step 2: Get ground truth relevance for this query
        query_relevance = self.relevance_lookup.get(query_id, {})
        relevant_doc_ids = [doc_id for doc_id, score in query_relevance.items() if score > 0]
        
        # Step 3: Calculate metrics for each k value
        query_metrics = {}
        
        for k in self.k_values:
            if k <= len(retrieved_doc_ids):
                # Calculate standard metrics using your existing EvaluationMetrics class
                precision_k = EvaluationMetrics.precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
                recall_k = EvaluationMetrics.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
                ndcg_k = EvaluationMetrics.ndcg_at_k(retrieved_doc_ids, query_relevance, k)
                
                query_metrics[f'precision@{k}'] = precision_k
                query_metrics[f'recall@{k}'] = recall_k
                query_metrics[f'ndcg@{k}'] = ndcg_k
        
        # Step 4: Calculate MRR
        mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        query_metrics['mrr'] = mrr
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'retrieved_docs': retrieved_doc_ids,
            'relevant_docs': relevant_doc_ids,
            'num_relevant_found': len([doc for doc in retrieved_doc_ids if doc in relevant_doc_ids]),
            'metrics': query_metrics
        }
    
    def evaluate_system_top_k(self, retrieval_system: RetrievalSystemInterface,
                             max_queries: Optional[int] = None,
                             verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a retrieval system on the full dataset to get top-k performance.
        
        Args:
            retrieval_system: System implementing RetrievalSystemInterface
            max_queries: Maximum number of queries to evaluate (None for all)
            verbose: Whether to print progress
            
        Returns:
            Complete evaluation results with aggregate metrics
        """
        start_time = time.time()
        
        # Get queries to evaluate
        queries_to_eval = self.dataset.queries.copy()
        if max_queries:
            queries_to_eval = queries_to_eval.head(max_queries)
        
        if verbose:
            print(f"\nEvaluating system on {len(queries_to_eval)} queries...")
        
        # Evaluate each query
        all_query_results = []
        all_metrics = []
        
        for i, (_, query_row) in enumerate(queries_to_eval.iterrows()):
            query_id = str(query_row['id'])
            query_text = str(query_row['text'])
            
            if verbose and (i + 1) % 25 == 0:
                print(f"  Progress: {i + 1}/{len(queries_to_eval)} queries")
            
            try:
                # Evaluate this query
                query_result = self.evaluate_query_top_k(query_id, query_text, retrieval_system)
                all_query_results.append(query_result)
                all_metrics.append(query_result['metrics'])
                
            except Exception as e:
                if verbose:
                    print(f"  Error evaluating query {query_id}: {e}")
                continue
        
        # Calculate aggregate metrics across all queries
        aggregate_metrics = self._calculate_aggregate_metrics(all_metrics)
        
        evaluation_time = time.time() - start_time
        
        results = {
            'aggregate_metrics': aggregate_metrics,
            'query_results': all_query_results,
            'evaluation_stats': {
                'num_queries_evaluated': len(all_query_results),
                'total_queries_available': len(self.dataset.queries),
                'evaluation_time_seconds': evaluation_time,
                'queries_per_second': len(all_query_results) / evaluation_time if evaluation_time > 0 else 0,
                'k_values_tested': self.k_values
            }
        }
        
        if verbose:
            print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
            print(f"Average Top-K Performance:")
            for metric, value in aggregate_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return results
    
    def _calculate_aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across all evaluated queries.
        
        Args:
            all_metrics: List of metric dictionaries from individual queries
            
        Returns:
            Dictionary with average metrics
        """
        if not all_metrics:
            return {}
        
        aggregate = {}
        
        # Calculate average for each metric
        for metric_name in all_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_metrics if metric_name in metrics]
            aggregate[metric_name] = sum(values) / len(values) if values else 0.0
        
        return aggregate
    
    def compare_systems_top_k(self, systems: Dict[str, RetrievalSystemInterface],
                             max_queries: Optional[int] = None,
                             verbose: bool = False) -> pd.DataFrame:
        """
        Compare multiple retrieval systems on top-k performance.
        
        Args:
            systems: Dictionary mapping system names to retrieval systems
            max_queries: Maximum queries to evaluate
            verbose: Whether to print progress
            
        Returns:
            DataFrame comparing system performance
        """
        if verbose:
            print(f"Comparing {len(systems)} systems on top-k evaluation...")
        
        comparison_results = []
        
        for system_name, system in systems.items():
            if verbose:
                print(f"\nEvaluating system: {system_name}")
            
            # Evaluate this system
            results = self.evaluate_system_top_k(system, max_queries, verbose)
            
            # Create comparison row
            row = {'system': system_name}
            row.update(results['aggregate_metrics'])
            row.update({
                'num_queries': results['evaluation_stats']['num_queries_evaluated'],
                'eval_time_sec': results['evaluation_stats']['evaluation_time_seconds']
            })
            
            comparison_results.append(row)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if verbose:
            print(f"\n{'='*60}")
            print("TOP-K COMPARISON RESULTS")
            print('='*60)
            
            # Show key metrics
            key_metrics = ['system'] + [col for col in comparison_df.columns 
                                      if any(metric in col for metric in ['precision@', 'recall@', 'ndcg@', 'mrr'])]
            print(comparison_df[key_metrics].round(4))
        
        return comparison_df
    
    def get_sample_evaluation(self, retrieval_system: RetrievalSystemInterface) -> Dict[str, Any]:
        """
        Get a sample evaluation on one query for testing/debugging.
        
        Args:
            retrieval_system: System to test
            
        Returns:
            Sample evaluation result
        """
        sample_query = self.dataset.get_sample_query()
        if not sample_query:
            return {"error": "No sample query available"}
        
        result = self.evaluate_query_top_k(
            sample_query['id'], 
            sample_query['text'], 
            retrieval_system
        )
        
        # Add ground truth info for comparison
        result['ground_truth'] = sample_query
        
        return result


# Example usage and testing
def demo_marco_top_k_evaluation():
    """Demonstrate how to use the Marco Top-K Evaluator."""
    
    print("=== MS MARCO Top-K Evaluation Demo ===")
    
    # Step 1: Load MS MARCO dataset
    print("\n1. Loading MS MARCO dataset...")
    dataset = MSMarcoDataset.create(limit=100)  # Small subset for demo
    if dataset is None:
        print("Failed to load MS MARCO dataset")
        return
    
    # Step 2: Initialize evaluator
    print("\n2. Initializing evaluator...")
    evaluator = MarcoTopKEvaluator(dataset, k_values=[1, 3, 5, 10])
    
    # Step 3: Create a simple mock retrieval system for testing
    class MockRetrievalSystem:
        def __init__(self, docs_df):
            self.all_doc_ids = docs_df['id'].astype(str).tolist()
        
        def retrieve(self, query: str, top_k: int) -> List[str]:
            # Simple mock: return first top_k documents
            # In practice, this would be your actual retrieval logic
            return self.all_doc_ids[:top_k]
    
    mock_system = MockRetrievalSystem(dataset.docs)
    
    # Step 4: Test sample evaluation
    print("\n3. Testing sample evaluation...")
    sample_result = evaluator.get_sample_evaluation(mock_system)
    print(f"Sample query: {sample_result.get('query_text', 'N/A')}")
    print(f"Sample metrics: {sample_result.get('metrics', {})}")
    
    # Step 5: Run full evaluation
    print("\n4. Running full system evaluation...")
    full_results = evaluator.evaluate_system_top_k(mock_system, max_queries=50, verbose=True)
    
    print("\nDemo completed! Replace MockRetrievalSystem with your actual retrieval system.")
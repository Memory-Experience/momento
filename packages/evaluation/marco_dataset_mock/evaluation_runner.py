"""
MS MARCO Retrieval System Evaluation

Implementation for evaluating retrieval systems on MS MARCO dataset.
"""

import random
import pandas as pd
from typing import List, Dict
from collections import Counter
from .marco_dataset import MSMarcoDataset
from .metrics import EvaluationMetrics
from .evaluator import MarcoTopKEvaluator


class MarcoRetrievalSystem:
    """Retrieval system implementation for MS MARCO evaluation."""
    
    def __init__(self, docs_df: pd.DataFrame, method: str = "tfidf"):
        self.docs_df = docs_df
        self.method = method
        self.doc_index = {}
        self.term_index = {}
        self.doc_lengths = {}
        self._build_index()
    
    def _build_index(self):
        """Build inverted index for retrieval."""
        print(f"Building {self.method} index...")
        
        for _, row in self.docs_df.iterrows():
            doc_id = str(row['id'])
            doc_text = str(row.get('content', row.get('text', '')))
            
            self.doc_index[doc_id] = doc_text
            tokens = self._preprocess_text(doc_text)
            self.doc_lengths[doc_id] = len(tokens)
            
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token not in self.term_index:
                    self.term_index[token] = {}
                self.term_index[token][doc_id] = count
        
        print(f"Index built: {len(self.term_index)} unique terms, {len(self.doc_index)} documents")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Basic text preprocessing."""
        return text.lower().split()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve top-k documents for query."""
        if self.method == "random":
            return self._random_retrieval(top_k)
        elif self.method == "bm25":
            return self._bm25_retrieval(query, top_k)
        else:
            return self._tfidf_retrieval(query, top_k)
    
    def _tfidf_retrieval(self, query: str, top_k: int) -> List[str]:
        """TF-IDF retrieval implementation."""
        query_tokens = self._preprocess_text(query)
        doc_scores = {}
        
        for token in query_tokens:
            if token in self.term_index:
                df = len(self.term_index[token])
                idf = len(self.doc_index) / (df + 1)
                
                for doc_id, tf in self.term_index[token].items():
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += tf * idf
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    def _bm25_retrieval(self, query: str, top_k: int) -> List[str]:
        """BM25 retrieval implementation."""
        query_tokens = self._preprocess_text(query)
        doc_scores = {}
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        k1, b = 1.5, 0.75
        
        for token in query_tokens:
            if token in self.term_index:
                df = len(self.term_index[token])
                idf = (len(self.doc_index) - df + 0.5) / (df + 0.5)
                
                for doc_id, tf in self.term_index[token].items():
                    doc_length = self.doc_lengths[doc_id]
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += score
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    def _random_retrieval(self, top_k: int) -> List[str]:
        """Random baseline retrieval."""
        doc_ids = list(self.doc_index.keys())
        return random.sample(doc_ids, min(top_k, len(doc_ids)))


def evaluate_multiple_queries():
    """Evaluate retrieval systems on multiple queries for robust testing."""
    
    print("Loading MS MARCO dataset...")
    dataset = MSMarcoDataset.create("msmarco-passage/dev/small", limit=1000)
    if dataset is None:
        print("Failed to load dataset")
        return
    
    print(f"Dataset loaded: {len(dataset.docs)} docs, {len(dataset.queries)} queries, {len(dataset.qrels)} qrels")
    
    # Create retrieval systems
    print("Creating retrieval systems...")
    tfidf_system = MarcoRetrievalSystem(dataset.docs, method="tfidf")
    bm25_system = MarcoRetrievalSystem(dataset.docs, method="bm25")
    random_system = MarcoRetrievalSystem(dataset.docs, method="random")
    
    # Find queries with relevant documents
    valid_queries = []
    for _, query_row in dataset.queries.iterrows():
        query_id = str(query_row['id'])
        query_text = str(query_row['text'])
        
        relevant_qrels = dataset.qrels[dataset.qrels['query_id'].astype(str) == query_id]
        if not relevant_qrels.empty:
            relevant_docs = relevant_qrels['doc_id'].astype(str).tolist()
            relevance_scores = dict(zip(
                relevant_qrels['doc_id'].astype(str),
                relevant_qrels['relevance']
            ))
            
            valid_queries.append({
                'id': query_id,
                'text': query_text,
                'relevant_docs': relevant_docs,
                'relevance_scores': relevance_scores
            })
            
            if len(valid_queries) >= 10:  # Test on 10 queries
                break
    
    if not valid_queries:
        print("No valid queries found with relevant documents")
        return
    
    print(f"\n{'='*80}")
    print(f"QUERIES BEING TESTED ({len(valid_queries)} queries)")
    print(f"{'='*80}")
    
    for i, query in enumerate(valid_queries, 1):
        print(f"Query {i}: {query['text']}")
        print(f"  ID: {query['id']}")
        print(f"  Relevant docs: {len(query['relevant_docs'])} documents")
        print(f"  Relevant doc IDs: {query['relevant_docs'][:3]}{'...' if len(query['relevant_docs']) > 3 else ''}")
        print()
    
    # Evaluate systems
    systems = [
        ('TF-IDF', tfidf_system),
        ('BM25', bm25_system),
        ('Random', random_system)
    ]
    
    # Collect results for each system
    system_results = {}
    
    print(f"{'='*80}")
    print("DETAILED RESULTS FOR EACH QUERY")
    print(f"{'='*80}")
    
    for system_name, system in systems:
        print(f"\n{system_name} SYSTEM:")
        print("-" * 40)
        
        all_metrics = {
            'precision@1': [], 'precision@3': [], 'precision@5': [],
            'recall@1': [], 'recall@3': [], 'recall@5': [],
            'ndcg@1': [], 'ndcg@3': [], 'ndcg@5': [],
            'mrr': []
        }
        
        for i, query in enumerate(valid_queries):
            retrieved_docs = system.retrieve(query['text'], top_k=10)
            
            # Calculate metrics
            p1 = EvaluationMetrics.precision_at_k(retrieved_docs, query['relevant_docs'], 1)
            p3 = EvaluationMetrics.precision_at_k(retrieved_docs, query['relevant_docs'], 3)
            p5 = EvaluationMetrics.precision_at_k(retrieved_docs, query['relevant_docs'], 5)
            
            r1 = EvaluationMetrics.recall_at_k(retrieved_docs, query['relevant_docs'], 1)
            r3 = EvaluationMetrics.recall_at_k(retrieved_docs, query['relevant_docs'], 3)
            r5 = EvaluationMetrics.recall_at_k(retrieved_docs, query['relevant_docs'], 5)
            
            n1 = EvaluationMetrics.ndcg_at_k(retrieved_docs, query['relevance_scores'], 1)
            n3 = EvaluationMetrics.ndcg_at_k(retrieved_docs, query['relevance_scores'], 3)
            n5 = EvaluationMetrics.ndcg_at_k(retrieved_docs, query['relevance_scores'], 5)
            
            mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, query['relevant_docs'])
            
            # Store metrics
            all_metrics['precision@1'].append(p1)
            all_metrics['precision@3'].append(p3)
            all_metrics['precision@5'].append(p5)
            all_metrics['recall@1'].append(r1)
            all_metrics['recall@3'].append(r3)
            all_metrics['recall@5'].append(r5)
            all_metrics['ndcg@1'].append(n1)
            all_metrics['ndcg@3'].append(n3)
            all_metrics['ndcg@5'].append(n5)
            all_metrics['mrr'].append(mrr)
            
            # Show which relevant docs were found
            relevant_found = [doc for doc in retrieved_docs[:5] if doc in query['relevant_docs']]
            
            print(f"Q{i+1}: {query['text'][:50]}...")
            print(f"    Retrieved: {retrieved_docs[:3]}")
            print(f"    Relevant found: {relevant_found if relevant_found else 'None'}")
            print(f"    P@1={p1:.3f}, P@3={p3:.3f}, R@3={r3:.3f}, MRR={mrr:.3f}")
            print()
        
        # Calculate averages
        avg_metrics = {metric: sum(values) / len(values) for metric, values in all_metrics.items()}
        system_results[system_name] = avg_metrics
    
    # Display final results
    print("="*80)
    print("FINAL AVERAGE PERFORMANCE ACROSS ALL QUERIES")
    print("="*80)
    
    print(f"{'System':<10} {'P@1':<6} {'P@3':<6} {'P@5':<6} {'R@1':<6} {'R@3':<6} {'R@5':<6} {'NDCG@3':<8} {'MRR':<6}")
    print("-" * 80)
    
    for system_name in ['TF-IDF', 'BM25', 'Random']:
        metrics = system_results[system_name]
        print(f"{system_name:<10} "
              f"{metrics['precision@1']:<6.3f} "
              f"{metrics['precision@3']:<6.3f} "
              f"{metrics['precision@5']:<6.3f} "
              f"{metrics['recall@1']:<6.3f} "
              f"{metrics['recall@3']:<6.3f} "
              f"{metrics['recall@5']:<6.3f} "
              f"{metrics['ndcg@3']:<8.3f} "
              f"{metrics['mrr']:<6.3f}")
    
    # Summary of which queries worked best
    print("\n" + "="*80)
    print("QUERY PERFORMANCE SUMMARY")
    print("="*80)
    
    print("Queries where TF-IDF found relevant documents:")
    tfidf_successful = []
    for i, query in enumerate(valid_queries):
        retrieved_docs = tfidf_system.retrieve(query['text'], top_k=5)
        relevant_found = [doc for doc in retrieved_docs if doc in query['relevant_docs']]
        if relevant_found:
            tfidf_successful.append(f"  Q{i+1}: {query['text'][:60]}... (found: {relevant_found})")
    
    if tfidf_successful:
        for success in tfidf_successful:
            print(success)
    else:
        print("  None - TF-IDF did not find relevant documents for any query")
    
    print("\nQueries where BM25 found relevant documents:")
    bm25_successful = []
    for i, query in enumerate(valid_queries):
        retrieved_docs = bm25_system.retrieve(query['text'], top_k=5)
        relevant_found = [doc for doc in retrieved_docs if doc in query['relevant_docs']]
        if relevant_found:
            bm25_successful.append(f"  Q{i+1}: {query['text'][:60]}... (found: {relevant_found})")
    
    if bm25_successful:
        for success in bm25_successful:
            print(success)
    else:
        print("  None - BM25 did not find relevant documents for any query")
    
    return system_results, valid_queries


def test_specific_query():
    """Test a specific query to verify evaluation is working correctly."""
    
    print("\n" + "="*80)
    print("TESTING SPECIFIC QUERY")
    print("="*80)
    
    dataset = MSMarcoDataset.create("msmarco-passage/dev/small", limit=200)
    if dataset is None:
        print("Failed to load dataset")
        return
    
    sample_query = dataset.get_sample_query()
    if not sample_query:
        print("No valid query found")
        return
    
    print("QUERY DETAILS:")
    print(f"  Query ID: {sample_query['id']}")
    print(f"  Query text: {sample_query['text']}")
    print(f"  Number of relevant documents: {len(sample_query['relevant_docs'])}")
    print(f"  Relevant document IDs: {sample_query['relevant_docs']}")
    print(f"  Relevance scores: {sample_query['relevance_scores']}")
    
    print("\nRELEVANT DOCUMENT CONTENT:")
    for doc_id in sample_query['relevant_docs'][:2]:
        doc_row = dataset.docs[dataset.docs['id'].astype(str) == doc_id]
        if not doc_row.empty:
            content = str(doc_row.iloc[0]['content'])[:200]
            print(f"  Doc {doc_id}: {content}...")
    
    # Create systems
    tfidf_system = MarcoRetrievalSystem(dataset.docs, method="tfidf")
    bm25_system = MarcoRetrievalSystem(dataset.docs, method="bm25")
    
    # Test retrieval
    systems = [('TF-IDF', tfidf_system), ('BM25', bm25_system)]
    
    print("\nRETRIEVAL RESULTS:")
    for system_name, system in systems:
        retrieved_docs = system.retrieve(sample_query['text'], top_k=10)
        
        # Calculate metrics
        p1 = EvaluationMetrics.precision_at_k(retrieved_docs, sample_query['relevant_docs'], 1)
        p3 = EvaluationMetrics.precision_at_k(retrieved_docs, sample_query['relevant_docs'], 3)
        mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, sample_query['relevant_docs'])
        
        # Find relevant documents in results
        relevant_found = []
        for i, doc_id in enumerate(retrieved_docs[:10]):
            if doc_id in sample_query['relevant_docs']:
                relevant_found.append((doc_id, i+1))  # doc_id and rank
        
        print(f"\n{system_name} System:")
        print(f"  Top 5 retrieved: {retrieved_docs[:5]}")
        if relevant_found:
            print(f"  Relevant documents found: {[f'{doc}@rank{rank}' for doc, rank in relevant_found]}")
        else:
            print(f"  Relevant documents found: None")
        print(f"  Precision@1: {p1:.3f}")
        print(f"  Precision@3: {p3:.3f}")
        print(f"  MRR: {mrr:.3f}")
        
        print(f"  Content of top retrieved docs:")
        for i, doc_id in enumerate(retrieved_docs[:3]):
            doc_row = dataset.docs[dataset.docs['id'].astype(str) == doc_id]
            if not doc_row.empty:
                content = str(doc_row.iloc[0]['content'])[:100]
                relevance = "RELEVANT" if doc_id in sample_query['relevant_docs'] else "not relevant"
                print(f"    {i+1}. Doc {doc_id} ({relevance}): {content}...")


def evaluate_with_marco_evaluator():
    """Evaluate using the existing MarcoTopKEvaluator for comparison."""
    
    print("\n" + "="*80)
    print("EVALUATION USING EXISTING MARCO EVALUATOR")
    print("="*80)
    
    dataset = MSMarcoDataset.create("msmarco-passage/dev/small", limit=500)
    if dataset is None:
        print("Failed to load dataset")
        return
    
    if not hasattr(dataset, 'get_name'):
        dataset.get_name = lambda: "MS MARCO (dev/small, limit=500)"
    
    evaluator = MarcoTopKEvaluator(dataset, k_values=[1, 3, 5, 10])
    
    tfidf_system = MarcoRetrievalSystem(dataset.docs, method="tfidf")
    bm25_system = MarcoRetrievalSystem(dataset.docs, method="bm25")
    
    print("Evaluating TF-IDF system...")
    tfidf_results = evaluator.evaluate_system_top_k(tfidf_system, max_queries=50, verbose=False)
    
    print("Evaluating BM25 system...")
    bm25_results = evaluator.evaluate_system_top_k(bm25_system, max_queries=50, verbose=False)
    
    print("\nTF-IDF Results:")
    for metric, value in tfidf_results['aggregate_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nBM25 Results:")
    for metric, value in bm25_results['aggregate_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return tfidf_results, bm25_results


def test_specific_query():
    """Test a specific query to verify evaluation is working correctly."""
    
    print("\n" + "="*80)
    print("TESTING SPECIFIC QUERY")
    print("="*80)
    
    dataset = MSMarcoDataset.create("msmarco-passage/dev/small", limit=200)
    if dataset is None:
        print("Failed to load dataset")
        return
    
    # Get first valid query
    sample_query = dataset.get_sample_query()
    if not sample_query:
        print("No valid query found")
        return
    
    print(f"Query: {sample_query['text']}")
    print(f"Relevant documents: {sample_query['relevant_docs']}")
    
    # Create systems
    tfidf_system = MarcoRetrievalSystem(dataset.docs, method="tfidf")
    bm25_system = MarcoRetrievalSystem(dataset.docs, method="bm25")
    
    # Test retrieval
    systems = [('TF-IDF', tfidf_system), ('BM25', bm25_system)]
    
    for system_name, system in systems:
        retrieved_docs = system.retrieve(sample_query['text'], top_k=10)
        
        # Calculate metrics
        p1 = EvaluationMetrics.precision_at_k(retrieved_docs, sample_query['relevant_docs'], 1)
        p3 = EvaluationMetrics.precision_at_k(retrieved_docs, sample_query['relevant_docs'], 3)
        mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, sample_query['relevant_docs'])
        
        # Find relevant documents in results
        relevant_found = [doc for doc in retrieved_docs[:5] if doc in sample_query['relevant_docs']]
        
        print(f"\n{system_name} System:")
        print(f"  Retrieved: {retrieved_docs[:5]}")
        print(f"  Relevant found: {relevant_found}")
        print(f"  Precision@1: {p1:.3f}")
        print(f"  Precision@3: {p3:.3f}")
        print(f"  MRR: {mrr:.3f}")


def main():
    """Main evaluation function."""
    
    print("MS MARCO Retrieval System Evaluation")
    print("="*80)
    
    print("TEST 1: Multiple Queries Evaluation")
    try:
        evaluate_multiple_queries()
    except Exception as e:
        print(f"Error in multiple queries evaluation: {e}")
    
    print("\nTEST 2: Marco Evaluator Comparison")
    try:
        evaluate_with_marco_evaluator()
    except Exception as e:
        print(f"Error in marco evaluator: {e}")
    
    print("\nTEST 3: Specific Query Test")
    try:
        test_specific_query()
    except Exception as e:
        print(f"Error in specific query test: {e}")
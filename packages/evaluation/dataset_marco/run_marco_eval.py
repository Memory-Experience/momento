import pandas as pd
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Try relative imports first (when run as module)
    from .dataframe_dataset import DataFrameDataset
    from .prepare_ms_marco import MSMarcoDataset
    from .metrics import precision_at_k, recall_at_k, ndcg_at_k, reciprocal_rank
except ImportError:
    # Fall back to direct imports (when run as script)
    from dataframe_dataset import DataFrameDataset
    from prepare_ms_marco import MSMarcoDataset
    from metrics import precision_at_k, recall_at_k, ndcg_at_k, reciprocal_rank

def convert_ms_marco_to_dataframes(ms_marco_dataset: MSMarcoDataset, limit: int = 1000):
    """Convert MS MARCO dataset to pandas DataFrames for evaluation.
    
    Args:
        ms_marco_dataset: The MS MARCO dataset instance
        limit: Maximum number of items to process (for speed)
    
    Returns:
        tuple: (docs_df, queries_df, qrels_df)
    """
    print(f"Converting MS MARCO to DataFrames (limit: {limit})...")
    
    # Convert documents
    docs_data = []
    for i, doc in enumerate(ms_marco_dataset.docs_iter()):
        if i >= limit:
            break
        docs_data.append({
            'id': doc['id'],
            'content': doc['content']
        })
    docs_df = pd.DataFrame(docs_data)
    
    # Convert queries  
    queries_data = []
    for i, query in enumerate(ms_marco_dataset.queries_iter()):
        if i >= limit:
            break
        queries_data.append({
            'id': query['id'],
            'text': query['text']
        })
    queries_df = pd.DataFrame(queries_data)
    
    # Convert qrels
    qrels_data = []
    for i, qrel in enumerate(ms_marco_dataset.qrels_iter()):
        if i >= limit:
            break
        qrels_data.append({
            'query_id': qrel['query_id'],
            'doc_id': qrel['doc_id'],
            'relevance': qrel['relevance']
        })
    qrels_df = pd.DataFrame(qrels_data)
    
    print(f"✅ Converted: {len(docs_df)} docs, {len(queries_df)} queries, {len(qrels_df)} qrels")
    return docs_df, queries_df, qrels_df

def evaluate_retrieval_metrics(dataset: DataFrameDataset, sample_size: int = 100):
    """Evaluate retrieval metrics on the dataset.
    
    Args:
        dataset: DataFrameDataset instance
        sample_size: Number of queries to evaluate
    
    Returns:
        dict: Evaluation results
    """
    print(f"🔍 Evaluating retrieval metrics on {sample_size} queries...")
    
    results = {
        'precision@1': [], 'precision@3': [], 'precision@5': [], 'precision@10': [],
        'recall@1': [], 'recall@3': [], 'recall@5': [], 'recall@10': [],
        'ndcg@1': [], 'ndcg@3': [], 'ndcg@5': [], 'ndcg@10': [],
        'mrr': []
    }
    
    # Sample queries for evaluation
    sample_queries = dataset.queries.head(sample_size)
    
    for _, query_row in sample_queries.iterrows():
        query_id = query_row['id']
        query_text = query_row['text']
        
        # Get relevant documents for this query
        relevant_qrels = dataset.qrels[dataset.qrels['query_id'] == query_id]
        gold_ids = set(relevant_qrels['doc_id'].tolist())
        
        if not gold_ids:
            continue  # Skip queries with no relevant docs
        
        # REALISTIC RETRIEVAL SIMULATION using text similarity
        # This simulates what a real retrieval system would do
        
        # Calculate similarity scores for all documents
        query_words = set(query_text.lower().split())
        doc_scores = []
        
        for _, doc_row in dataset.docs.iterrows():
            doc_id = doc_row['id']
            doc_content = doc_row['content'].lower()
            doc_words = set(doc_content.split())
            
            # Simple TF-IDF-like scoring: word overlap + length normalization
            overlap = len(query_words.intersection(doc_words))
            doc_length = len(doc_words)
            query_length = len(query_words)
            
            if doc_length > 0 and query_length > 0:
                # Jaccard similarity with length bias
                similarity = overlap / (query_length + doc_length - overlap + 1)
                # Add small random noise to break ties
                import random
                similarity += random.uniform(0, 0.01)
            else:
                similarity = 0.0
            
            doc_scores.append((doc_id, similarity))
        
        # Sort by similarity score (highest first) - this is realistic ranking
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [doc_id for doc_id, _ in doc_scores[:20]]  # Top 20 results
        
        # Calculate metrics for different k values
        for k in [1, 3, 5, 10]:
            if k <= len(ranked_ids):
                results[f'precision@{k}'].append(precision_at_k(ranked_ids, gold_ids, k))
                results[f'recall@{k}'].append(recall_at_k(ranked_ids, gold_ids, k))
                results[f'ndcg@{k}'].append(ndcg_at_k(ranked_ids, gold_ids, k))
        
        # MRR
        results['mrr'].append(reciprocal_rank(ranked_ids, gold_ids))
    
    # Calculate averages
    avg_results = {}
    for metric, values in results.items():
        if values:
            avg_results[metric] = sum(values) / len(values)
        else:
            avg_results[metric] = 0.0
    
    return avg_results

def run_marco_eval():
    """Main function to run MS MARCO evaluation with retrieval metrics only."""
    print("🚀 Starting MS MARCO Retrieval Evaluation")
    print("=" * 50)
    
    # Load MS MARCO dataset (small variant for speed)
    print("📥 Loading MS MARCO dataset...")
    ms_marco = MSMarcoDataset("msmarco-passage/dev/small")
    
    # Convert to DataFrames (limit for speed)
    docs_df, queries_df, qrels_df = convert_ms_marco_to_dataframes(ms_marco, limit=1000)
    
    # Create DataFrameDataset
    dataset = DataFrameDataset(docs_df, queries_df, qrels_df)
    print(f"Dataset ready: {len(dataset.docs)} docs, {len(dataset.queries)} queries")
    
    # Evaluate retrieval metrics only
    print("\n" + "="*40)
    retrieval_results = evaluate_retrieval_metrics(dataset, sample_size=100)
    
    print("RETRIEVAL METRICS RESULTS:")
    print("-" * 40)
    for metric, value in retrieval_results.items():
        print(f"   {metric:15s}: {value:.4f}")
    
    print("\nMS MARCO retrieval evaluation completed!")
    
    return {
        'retrieval': retrieval_results,
        'dataset_info': {
            'docs': len(dataset.docs),
            'queries': len(dataset.queries),
            'qrels': len(dataset.qrels)
        }
    }

if __name__ == "__main__":
    # Run the evaluation when script is executed directly
    results = run_marco_eval()
    print(f"\n Final Results Summary:")
    print(f"   Dataset: {results['dataset_info']}")
    print(f"   Metrics: {len(results['retrieval'])} computed")


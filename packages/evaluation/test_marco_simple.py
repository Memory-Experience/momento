"""
Simple test script for MS MARCO retrieval evaluation.
This avoids complex import issues by running directly.
"""

import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Direct imports
from dataset_marco.dataframe_dataset import DataFrameDataset
from dataset_marco.prepare_ms_marco import MSMarcoDataset
from dataset_marco.metrics import precision_at_k, recall_at_k, ndcg_at_k, reciprocal_rank
import pandas as pd

def simple_marco_test():
    """Simple test of MS MARCO retrieval evaluation."""
    print("🧪 Simple MS MARCO Retrieval Test")
    print("=" * 40)
    
    try:
        # Load small MS MARCO dataset
        print("📥 Loading MS MARCO (small)...")
        ms_marco = MSMarcoDataset("msmarco-passage/dev/small")
        
        # Convert just a few samples for testing
        print("🔄 Converting to DataFrames...")
        docs_data = []
        queries_data = []
        qrels_data = []
        
        # Get first 10 documents
        for i, doc in enumerate(ms_marco.docs_iter()):
            if i >= 10:
                break
            docs_data.append({
                'id': doc['id'],
                'content': doc['content']
            })
        
        # Get first 5 queries
        for i, query in enumerate(ms_marco.queries_iter()):
            if i >= 5:
                break
            queries_data.append({
                'id': query['id'],
                'text': query['text']
            })
        
        # Get first 10 qrels
        for i, qrel in enumerate(ms_marco.qrels_iter()):
            if i >= 10:
                break
            qrels_data.append({
                'query_id': qrel['query_id'],
                'doc_id': qrel['doc_id'],
                'relevance': qrel['relevance']
            })
        
        # Create DataFrames
        docs_df = pd.DataFrame(docs_data)
        queries_df = pd.DataFrame(queries_data)
        qrels_df = pd.DataFrame(qrels_data)
        
        print(f"✅ Data loaded: {len(docs_df)} docs, {len(queries_df)} queries, {len(qrels_df)} qrels")
        
        # Create DataFrameDataset
        dataset = DataFrameDataset(docs_df, queries_df, qrels_df)
        
        # Test one query evaluation
        if len(dataset.queries) > 0 and len(dataset.qrels) > 0:
            first_query = dataset.queries.iloc[0]
            query_id = first_query['id']
            
            # Get relevant docs for this query
            relevant_qrels = dataset.qrels[dataset.qrels['query_id'] == query_id]
            gold_ids = set(relevant_qrels['doc_id'].tolist())
            
            if gold_ids:
                # Create mock ranking
                all_doc_ids = dataset.docs['id'].tolist()
                ranked_ids = all_doc_ids[:5]  # Top 5 results
                
                # Calculate metrics
                p1 = precision_at_k(ranked_ids, gold_ids, 1)
                r1 = recall_at_k(ranked_ids, gold_ids, 1)
                ndcg1 = ndcg_at_k(ranked_ids, gold_ids, 1)
                mrr = reciprocal_rank(ranked_ids, gold_ids)
                
                print(f"📊 Sample Metrics for query '{first_query['text']}':")
                print(f"   Precision@1: {p1:.4f}")
                print(f"   Recall@1: {r1:.4f}")
                print(f"   NDCG@1: {ndcg1:.4f}")
                print(f"   MRR: {mrr:.4f}")
            else:
                print("⚠️ No relevant documents found for test query")
        
        print("\n✅ MS MARCO retrieval test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_marco_test()

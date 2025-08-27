"""
MS MARCO Data Structure Inspector

Quick script to understand the MS MARCO data structure and fix the evaluation demo.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)

def inspect_marco_data():
    """Inspect MS MARCO data structure."""
    print(" INSPECTING MS MARCO DATA STRUCTURE")
    print("=" * 50)
    
    try:
        from dataset_adapters.marco_adapter import create_ms_marco_adapter
    except ImportError as e:
        print(f" Import failed: {e}")
        return
    
    # Load small sample
    dataset = create_ms_marco_adapter(limit=50)
    if not dataset:
        print(" Failed to load dataset")
        return
    
    queries_df = dataset.get_queries()
    docs_df = dataset.get_documents()
    qrels_df = dataset.get_relevance_judgments()
    
    print(f" DATA OVERVIEW:")
    print(f"   Queries: {len(queries_df)}")
    print(f"   Documents: {len(docs_df)}")
    print(f"   Relevance judgments: {len(qrels_df)}")
    
    print(f"\n QUERIES SAMPLE:")
    print(queries_df.head(3))
    
    print(f"\n DOCUMENTS SAMPLE:")
    print(docs_df.head(3))
    
    print(f"\n QRELS SAMPLE:")
    print(qrels_df.head(3))
    
    # Check data types
    print(f"\n DATA TYPES:")
    print(f"   Query IDs type: {type(queries_df['id'].iloc[0])}")
    print(f"   Doc IDs type: {type(docs_df['id'].iloc[0])}")
    print(f"   QRel query_id type: {type(qrels_df['query_id'].iloc[0])}")
    print(f"   QRel doc_id type: {type(qrels_df['doc_id'].iloc[0])}")
    
    # Check for overlaps
    query_ids_in_qrels = set(qrels_df['query_id'].unique())
    query_ids_available = set(queries_df['id'].unique())
    doc_ids_in_qrels = set(qrels_df['doc_id'].unique())
    doc_ids_available = set(docs_df['id'].unique())
    
    print(f"\n🔄 DATA ALIGNMENT:")
    print(f"   Queries with relevance judgments: {len(query_ids_in_qrels & query_ids_available)}")
    print(f"   Documents referenced in qrels: {len(doc_ids_in_qrels)}")
    print(f"   Documents available: {len(doc_ids_available)}")
    print(f"   Documents overlap: {len(doc_ids_in_qrels & doc_ids_available)}")
    
    # Find a working example
    print(f"\n WORKING EXAMPLE:")
    for _, qrel in qrels_df.head(10).iterrows():
        query_id = qrel['query_id']
        doc_id = qrel['doc_id']
        
        query_match = queries_df[queries_df['id'] == query_id]
        doc_match = docs_df[docs_df['id'] == doc_id]
        
        if not query_match.empty and not doc_match.empty:
            query_text = query_match['text'].iloc[0]
            doc_content = doc_match['content'].iloc[0]
            
            print(f"   Query ID: {query_id}")
            print(f"   Query: '{query_text[:80]}...'")
            print(f"   Doc ID: {doc_id}")
            print(f"   Document: '{doc_content[:100]}...'")
            print(f"   Relevance: {qrel['relevance']}")
            break
    else:
        print("    No working examples found - data mismatch issue")

if __name__ == "__main__":
    inspect_marco_data()

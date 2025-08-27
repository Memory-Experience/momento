"""
MS MARCO Dataset Structure Examples for Testing DataFrameDataset

This file shows the exact structure of MS MARCO data to help test
your DataFrameDataset implementation.
"""

import pandas as pd
from pathlib import Path

# ==============================================================================
# MS MARCO DATASET STRUCTURE
# ==============================================================================

def create_marco_example_docs():
    """
    MS MARCO Documents Structure:
    - id: Document identifier (string)
    - content: Full document text (string)
    - title: Document title (optional, string)
    """
    docs_data = [
        {
            'id': 'doc_0',
            'content': 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.',
            'title': 'Introduction to Machine Learning'
        },
        {
            'id': 'doc_1', 
            'content': 'Python is an interpreted, high-level and general-purpose programming language. Python\'s design philosophy emphasizes code readability with its notable use of significant whitespace.',
            'title': 'Python Programming Language Overview'
        },
        {
            'id': 'doc_2',
            'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.',
            'title': 'Deep Learning Fundamentals'
        },
        {
            'id': 'doc_3',
            'content': 'Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.',
            'title': 'Natural Language Processing Basics'
        },
        {
            'id': 'doc_4',
            'content': 'Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources.',
            'title': 'Information Retrieval Systems'
        }
    ]
    return pd.DataFrame(docs_data)

def create_marco_example_queries():
    """
    MS MARCO Queries Structure:
    - id: Query identifier (string)  
    - text: Query text (string)
    """
    queries_data = [
        {'id': 'query_1', 'text': 'what is machine learning'},
        {'id': 'query_2', 'text': 'python programming language features'},
        {'id': 'query_3', 'text': 'difference between machine learning and deep learning'},
        {'id': 'query_4', 'text': 'natural language processing applications'},
        {'id': 'query_5', 'text': 'how does information retrieval work'}
    ]
    return pd.DataFrame(queries_data)

def create_marco_example_qrels():
    """
    MS MARCO QRELs (Query Relevance Judgments) Structure:
    - query_id: Query identifier (string)
    - doc_id: Document identifier (string) 
    - relevance: Relevance score (int, typically 0 or 1 for binary, or 0-3 for graded)
    
    Note: MS MARCO uses binary relevance (0=not relevant, 1=relevant)
    """
    qrels_data = [
        # Query 1: "what is machine learning" 
        {'query_id': 'query_1', 'doc_id': 'doc_0', 'relevance': 1},  # Direct match
        {'query_id': 'query_1', 'doc_id': 'doc_2', 'relevance': 1},  # Related (deep learning)
        
        # Query 2: "python programming language features"
        {'query_id': 'query_2', 'doc_id': 'doc_1', 'relevance': 1},  # Direct match
        
        # Query 3: "difference between machine learning and deep learning"
        {'query_id': 'query_3', 'doc_id': 'doc_0', 'relevance': 1},  # ML definition
        {'query_id': 'query_3', 'doc_id': 'doc_2', 'relevance': 1},  # Deep learning definition
        
        # Query 4: "natural language processing applications"
        {'query_id': 'query_4', 'doc_id': 'doc_3', 'relevance': 1},  # Direct match
        
        # Query 5: "how does information retrieval work"
        {'query_id': 'query_5', 'doc_id': 'doc_4', 'relevance': 1},  # Direct match
    ]
    return pd.DataFrame(qrels_data)

# ==============================================================================
# ACTUAL MS MARCO FORMAT EXAMPLES
# ==============================================================================

def show_real_marco_format():
    """
    Show what real MS MARCO data looks like:
    """
    print("🔍 REAL MS MARCO PASSAGE FORMAT:")
    print("=" * 50)
    
    print("\n📄 Documents (passages):")
    print("doc_id\tcontent")
    print("0\tThe presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives lost.")
    print("1\tThe Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada.")
    
    print("\n❓ Queries:")
    print("query_id\tquery")
    print("1048585\twhat was the manhattan project")
    print("1048586\twhen did the manhattan project start")
    
    print("\n🎯 QRELs (Query Relevance Judgments):")
    print("query_id\tdoc_id\trelevance")
    print("1048585\t1\t1")
    print("1048585\t0\t0")
    print("1048586\t1\t1")

# ==============================================================================
# TEST YOUR DATAFRAME_DATASET IMPLEMENTATION
# ==============================================================================

def test_dataframe_dataset():
    """Test your DataFrameDataset with MS MARCO structure."""
    try:
        # Import your implementation
        from dataset_marco.dataframe_dataset import DataFrameDataset
        
        print("🧪 TESTING DataFrameDataset with MS MARCO Structure")
        print("=" * 60)
        
        # Create test data
        docs_df = create_marco_example_docs()
        queries_df = create_marco_example_queries()
        qrels_df = create_marco_example_qrels()
        
        print(f"📊 Test Data Created:")
        print(f"   Documents: {len(docs_df)} items")
        print(f"   Queries: {len(queries_df)} items") 
        print(f"   QRELs: {len(qrels_df)} judgments")
        
        # Initialize dataset
        dataset = DataFrameDataset(docs_df, queries_df, qrels_df)
        print(f"\n✅ DataFrameDataset initialized successfully")
        
        # Test properties
        print(f"\n📈 Dataset Properties:")
        print(f"   docs property: {len(dataset.docs)} documents")
        print(f"   queries property: {len(dataset.queries)} queries")
        print(f"   qrels property: {len(dataset.qrels)} judgments")
        
        # Test iterators
        print(f"\n🔄 Testing Iterators:")
        docs_count = sum(1 for _ in dataset.docs_iter())
        queries_count = sum(1 for _ in dataset.queries_iter())
        qrels_count = sum(1 for _ in dataset.qrels_iter())
        
        print(f"   docs_iter(): {docs_count} items")
        print(f"   queries_iter(): {queries_count} items")
        print(f"   qrels_iter(): {qrels_count} items")
        
        # Test specific query
        print(f"\n🔍 Sample Query Test:")
        sample_query = dataset.queries.iloc[0]
        query_id = sample_query['id']
        query_text = sample_query['text']
        
        # Find relevant docs for this query
        relevant_docs = dataset.qrels[dataset.qrels['query_id'] == query_id]
        
        print(f"   Query: {query_text}")
        print(f"   Query ID: {query_id}")
        print(f"   Relevant docs: {len(relevant_docs)} found")
        
        for _, rel in relevant_docs.iterrows():
            doc_id = rel['doc_id']
            doc_content = dataset.docs[dataset.docs['id'] == doc_id]['content'].iloc[0]
            print(f"     - {doc_id}: {doc_content[:80]}...")
        
        print(f"\n✅ All tests passed! Your DataFrameDataset works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_marco_files():
    """Create sample CSV files in MS MARCO format for testing."""
    output_dir = Path("sample_marco_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save sample data as CSV
    docs_df = create_marco_example_docs()
    queries_df = create_marco_example_queries()
    qrels_df = create_marco_example_qrels()
    
    docs_df.to_csv(output_dir / "docs.csv", index=False)
    queries_df.to_csv(output_dir / "queries.csv", index=False)
    qrels_df.to_csv(output_dir / "qrels.csv", index=False)
    
    print(f"📁 Sample MS MARCO files created in: {output_dir}")
    print(f"   - docs.csv: {len(docs_df)} documents")
    print(f"   - queries.csv: {len(queries_df)} queries")
    print(f"   - qrels.csv: {len(qrels_df)} relevance judgments")

if __name__ == "__main__":
    # Show real MS MARCO format
    show_real_marco_format()
    
    print("\n" + "="*60)
    
    # Test your implementation
    test_dataframe_dataset()
    
    print("\n" + "="*60)
    
    # Create sample files
    create_sample_marco_files()

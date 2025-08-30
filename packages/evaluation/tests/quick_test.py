"""
First Query Evaluator for MS MARCO Dataset

Downloads MS MARCO data and evaluates using just the first query for quick testing.
"""

import pandas as pd
import requests
import io
from typing import Dict, Any, Optional, List
from ..marco_dataset_mock.marco_dataset import MSMarcoDataset
from ..marco_dataset_mock.metrics import EvaluationMetrics
from ..marco_dataset_mock.evaluator import MarcoTopKEvaluator
from ..marco_dataset_mock.dataset import DataFrameDataset


# Update the download_first_marco_query function with better debugging:

def download_first_marco_query() -> Optional[Dict[str, Any]]:
    """
    Download MS MARCO data and get the first query with its relevant documents.
    
    Returns:
        Dictionary with first query info and relevant documents
    """
    
    print("Downloading MS MARCO dev small data for first query...")
    
    try:
        # MS MARCO dev small URLs
        base_url = "https://msmarco.blob.core.windows.net/msmarcoranking/"
        
        # 1. Download qrels first (to get first relevant query-doc pair)
        print("1. Downloading qrels...")
        qrels_url = base_url + "qrels.dev.small.tsv"
        
        try:
            qrels_response = requests.get(qrels_url, timeout=30)
            qrels_response.raise_for_status()  # Raises an HTTPError for bad responses
            print(f"   Qrels downloaded successfully ({len(qrels_response.text)} chars)")
        except requests.exceptions.RequestException as e:
            print(f"   ERROR downloading qrels: {e}")
            return None
        
        first_qrel = None
        qrel_lines = qrels_response.text.strip().split('\n')
        print(f"   Processing {len(qrel_lines)} qrel lines...")
        
        for i, line in enumerate(qrel_lines[:10]):  # Check first 10 lines
            parts = line.split('\t')
            print(f"   Line {i}: {parts[:4] if len(parts) >= 4 else parts}")  # Debug
            
            if len(parts) >= 4:
                first_qrel = {
                    'query_id': parts[0],
                    'doc_id': parts[2], 
                    'relevance': int(parts[3])
                }
                print(f"   Found first qrel: {first_qrel}")
                break
        
        if not first_qrel:
            print("   ERROR: No valid qrels found!")
            print(f"   Sample lines: {qrel_lines[:3]}")
            return None
            
        # 2. Download queries to get the query text
        print("2. Downloading queries...")
        queries_url = base_url + "queries.dev.small.tsv"
        
        try:
            queries_response = requests.get(queries_url, timeout=30)
            queries_response.raise_for_status()
            print(f"   Queries downloaded successfully ({len(queries_response.text)} chars)")
        except requests.exceptions.RequestException as e:
            print(f"   ERROR downloading queries: {e}")
            return None
        
        query_text = None
        query_lines = queries_response.text.strip().split('\n')
        print(f"   Processing {len(query_lines)} query lines...")
        
        for i, line in enumerate(query_lines[:10]):  # Check first 10 lines
            parts = line.split('\t')
            print(f"   Query line {i}: {parts[:2] if len(parts) >= 2 else parts}")  # Debug
            
            if len(parts) >= 2 and parts[0] == first_qrel['query_id']:
                query_text = parts[1]
                print(f"   Found query text: {query_text}")
                break
        
        if not query_text:
            print(f"   ERROR: Could not find query text for ID {first_qrel['query_id']}")
            print(f"   Sample query lines: {query_lines[:3]}")
            return None
        
        # 3. Download collection to get document text (this is the big file)
        print("3. Downloading documents (this may take a moment)...")
        collection_url = base_url + "collection.tsv"
        
        target_doc_text = None
        sample_docs = {}
        doc_count = 0
        
        try:
            # Stream download to find our target document
            with requests.get(collection_url, stream=True, timeout=120) as response:
                response.raise_for_status()
                print(f"   Starting document stream download...")
                
                content = ""
                
                for chunk_num, chunk in enumerate(response.iter_content(chunk_size=8192, decode_unicode=True)):
                    if chunk_num % 1000 == 0:  # Progress indicator
                        print(f"   Processing chunk {chunk_num}, found {doc_count} docs so far...")
                    
                    content += chunk
                    lines = content.split('\n')
                    content = lines[-1]  # Keep incomplete line
                    
                    for line in lines[:-1]:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            doc_id, doc_content = parts[0], parts[1]
                            
                            # Save some sample documents
                            if doc_count < 50:  # Get 50 sample docs
                                sample_docs[doc_id] = doc_content
                            
                            # Check if this is our target document
                            if doc_id == first_qrel['doc_id']:
                                target_doc_text = doc_content
                                print(f"Found target document {doc_id}!")
                            
                            doc_count += 1
                            
                            # Stop after finding target and getting enough samples
                            if target_doc_text and doc_count >= 50:
                                print(f"   Stopping after finding target and {doc_count} sample docs")
                                break
                    
                    if target_doc_text and doc_count >= 50:
                        break
                        
        except requests.exceptions.RequestException as e:
            print(f"   ERROR downloading documents: {e}")
            return None
        
        print(f"   Document download complete. Processed {doc_count} documents")
        print(f"   Target doc found: {'Yes' if target_doc_text else 'No'}")
        print(f"   Sample docs collected: {len(sample_docs)}")
        
        if not target_doc_text:
            print(f"   WARNING: Could not find target document {first_qrel['doc_id']}")
            print(f"   Will use placeholder text")
            target_doc_text = f"Document {first_qrel['doc_id']} - content not found in collection"
        
        result = {
            'query_id': first_qrel['query_id'],
            'query_text': query_text,
            'relevant_doc_id': first_qrel['doc_id'],
            'relevant_doc_text': target_doc_text,
            'relevance_score': first_qrel['relevance'],
            'sample_documents': sample_docs
        }
        
        print(f"Successfully created result with {len(result)} fields")
        return result
        
    except Exception as e:
        print(f"UNEXPECTED ERROR in download_first_marco_query: {e}")
        import traceback
        traceback.print_exc()
        return None


# Also add a simple test function to check connectivity:

def test_marco_connectivity():
    """Test if we can reach MS MARCO URLs."""
    
    print("=== Testing MS MARCO Connectivity ===")
    
    base_url = "https://msmarco.blob.core.windows.net/msmarcoranking/"
    test_urls = [
        ("qrels", "qrels.dev.small.tsv"),
        ("queries", "queries.dev.small.tsv"),
        ("collection", "collection.tsv")
    ]
    
    for name, filename in test_urls:
        url = base_url + filename
        print(f"Testing {name}: {url}")
        
        try:
            response = requests.head(url, timeout=10)  # HEAD request to check availability
            print(f"  Status: {response.status_code}")
            if 'content-length' in response.headers:
                size_mb = int(response.headers['content-length']) / (1024 * 1024)
                print(f"  Size: {size_mb:.1f} MB")
            print(f"Available")
        except Exception as e:
            print(f" Error: {e}")
        
        print()


# Add this simple alternative that creates test data without downloading:

def create_simple_test_data():
    """Create simple test data without downloading."""
    
    print("=== Creating Simple Test Data (No Download) ===")
    
    # Create realistic test data
    test_data = {
        'query_id': 'test_001',
        'query_text': 'what is paula deen\'s brother',
        'relevant_doc_id': 'doc_relevant',
        'relevant_doc_text': 'Paula Deen\'s brother is Earl W. Hiers Jr., known as "Bubba." He has appeared on several of her cooking shows and is also involved in the restaurant business with Paula.',
        'relevance_score': 1,
        'sample_documents': {
            'doc_relevant': 'Paula Deen\'s brother is Earl W. Hiers Jr., known as "Bubba." He has appeared on several of her cooking shows.',
            'doc_001': 'Information about southern cooking and traditional recipes from Georgia.',
            'doc_002': 'Television cooking shows and celebrity chefs in American media.',
            'doc_003': 'Restaurant business and food industry in the southeastern United States.',
            'doc_004': 'Family members of television personalities and their career involvement.',
            'doc_005': 'Cooking techniques and ingredients commonly used in southern cuisine.',
        }
    }
    
    print(f"Created test data:")
    print(f"  Query: {test_data['query_text']}")
    print(f"  Relevant doc: {test_data['relevant_doc_id']}")
    print(f"  Sample docs: {len(test_data['sample_documents'])}")
    
    return test_data


def create_first_query_dataset(first_query_data: Dict[str, Any]) -> DataFrameDataset:
    """
    Create a minimal dataset with just the first query and related documents.
    
    Args:
        first_query_data: Data from download_first_marco_query()
        
    Returns:
        DataFrameDataset with minimal data for testing
    """
    
    print("Creating minimal dataset for first query evaluation...")
    
    # Create documents DataFrame
    docs_data = []
    
    # Add the relevant document first
    if first_query_data['relevant_doc_text']:
        docs_data.append({
            'id': first_query_data['relevant_doc_id'],
            'content': first_query_data['relevant_doc_text']
        })
    
    # Add sample documents
    for doc_id, doc_content in first_query_data['sample_documents'].items():
        if doc_id != first_query_data['relevant_doc_id']:  # Don't duplicate
            docs_data.append({
                'id': doc_id,
                'content': doc_content
            })
    
    docs_df = pd.DataFrame(docs_data)
    
    # Create queries DataFrame
    queries_df = pd.DataFrame([{
        'id': first_query_data['query_id'],
        'text': first_query_data['query_text']
    }])
    
    # Create qrels DataFrame
    qrels_df = pd.DataFrame([{
        'query_id': first_query_data['query_id'],
        'doc_id': first_query_data['relevant_doc_id'],
        'relevance': first_query_data['relevance_score']
    }])
    
    print(f"   Created dataset: {len(docs_df)} docs, {len(queries_df)} queries, {len(qrels_df)} qrels")
    
    return DataFrameDataset(docs_df, queries_df, qrels_df)


# Update the get_first_query_from_existing_dataset function:

def get_first_query_from_existing_dataset() -> Optional[Dict[str, Any]]:
    """
    Use existing MS MARCO dataset implementation to get the first query.
    
    Returns:
        Dictionary with first query info and relevant documents
    """
    
    print("Loading MS MARCO data using existing implementation...")
    
    try:
        # Use your existing MSMarcoDataset class
        dataset = MSMarcoDataset.create("msmarco-passage/dev/small", limit=100)
        if dataset is None:
            print("Failed to load dataset using existing implementation")
            return None
        
        print(f"Successfully loaded dataset with {len(dataset.docs)} docs, {len(dataset.queries)} queries")
        
        # Get the first query that has relevant documents
        for _, query_row in dataset.queries.iterrows():
            query_id = str(query_row['id'])
            query_text = str(query_row['text'])
            
            # Find relevant documents for this query
            relevant_qrels = dataset.qrels[dataset.qrels['query_id'].astype(str) == query_id]
            
            if not relevant_qrels.empty:
                # Found a query with relevant documents
                relevant_doc_ids = relevant_qrels['doc_id'].astype(str).tolist()
                relevance_scores = dict(zip(
                    relevant_qrels['doc_id'].astype(str), 
                    relevant_qrels['relevance']
                ))
                
                # Get the text of the first relevant document
                first_relevant_doc_id = relevant_doc_ids[0]
                relevant_doc_text = ""
                
                doc_row = dataset.docs[dataset.docs['id'].astype(str) == first_relevant_doc_id]
                if not doc_row.empty:
                    relevant_doc_text = str(doc_row.iloc[0]['content'])
                
                # Create sample documents dictionary
                sample_docs = {}
                for _, doc_row in dataset.docs.head(20).iterrows():  # Get first 20 docs
                    doc_id = str(doc_row['id'])
                    doc_content = str(doc_row['content'])
                    sample_docs[doc_id] = doc_content
                
                result = {
                    'query_id': query_id,
                    'query_text': query_text,
                    'relevant_doc_ids': relevant_doc_ids,
                    'relevant_doc_text': relevant_doc_text,
                    'relevance_scores': relevance_scores,
                    'sample_documents': sample_docs,
                    'original_dataset': dataset  # Keep the original dataset
                }
                
                print(f"Found first query with relevant docs:")
                print(f"  Query ID: {query_id}")
                print(f"  Query text: {query_text}")
                print(f"  Relevant docs: {relevant_doc_ids}")
                
                return result
        
        print("No queries with relevant documents found!")
        return None
        
    except Exception as e:
        print(f"Error using existing dataset implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


# Update evaluate_with_first_query to create DataFrameDataset with proper name:

def evaluate_with_first_query():
    """
    Main function to use existing MS MARCO data and evaluate using the first query.
    
    Returns:
        tuple: (dataset, first_query_data) or (None, None) if failed
    """
    
    print("=== First Query MS MARCO Evaluation (Using Existing Implementation) ===")
    
    try:
        # Step 1: Get first query using existing implementation
        first_query_data = get_first_query_from_existing_dataset()
        if not first_query_data:
            print("Failed to get first query data")
            return None, None
        
        # Step 2: Use the original dataset directly (it should work with MarcoTopKEvaluator)
        original_dataset = first_query_data['original_dataset']
        
        # Check if the original dataset has get_name method, if not add it
        if not hasattr(original_dataset, 'get_name'):
            # Add the method dynamically
            original_dataset.get_name = lambda: f"MS MARCO ({getattr(original_dataset, 'subset_name', 'dev/small')}, limit={getattr(original_dataset, 'limit', 100)})"
        
        dataset = original_dataset  # Use the original dataset
        
        print(f"Using dataset: {dataset.get_name()}")
        
        # Step 3: Create your retrieval systems
        from ms_macro_eval import MarcoRetrievalSystem
        
        print("\nCreating retrieval systems...")
        tfidf_system = MarcoRetrievalSystem(dataset.docs, method="tfidf")
        bm25_system = MarcoRetrievalSystem(dataset.docs, method="bm25")
        random_system = MarcoRetrievalSystem(dataset.docs, method="random")
        
        # Step 4: Test retrieval on the first query
        query_text = first_query_data['query_text']
        relevant_docs = first_query_data['relevant_doc_ids']
        relevance_scores = first_query_data['relevance_scores']
        
        print(f"\nEvaluating systems on first query:")
        print(f"Query: {query_text}")
        print(f"Relevant docs: {relevant_docs}")
        
        # Step 5: Evaluate each system
        systems = [
            ('TF-IDF', tfidf_system),
            ('BM25', bm25_system),
            ('Random', random_system)
        ]
        
        print(f"\n{'='*80}")
        print("FIRST QUERY EVALUATION RESULTS")
        print('='*80)
        print(f"{'System':<10} {'P@1':<6} {'P@3':<6} {'P@5':<6} {'R@3':<6} {'NDCG@3':<8} {'MRR':<6}")
        print("-" * 60)
        
        all_results = []
        
        for system_name, system in systems:
            # Get retrieval results
            retrieved_docs = system.retrieve(query_text, top_k=10)
            
            # Calculate metrics using your existing EvaluationMetrics class
            p1 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 1)
            p3 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 3)
            p5 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 5)
            r3 = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_docs, 3)
            ndcg3 = EvaluationMetrics.ndcg_at_k(retrieved_docs, relevance_scores, 3)
            mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_docs)
            
            print(f"{system_name:<10} {p1:<6.3f} {p3:<6.3f} {p5:<6.3f} {r3:<6.3f} {ndcg3:<8.3f} {mrr:<6.3f}")
            
            # Store results
            all_results.append({
                'system': system_name,
                'retrieved_docs': retrieved_docs[:5],
                'relevant_found': [doc for doc in retrieved_docs[:5] if doc in relevant_docs],
                'metrics': {
                    'precision@1': p1, 'precision@3': p3, 'precision@5': p5,
                    'recall@3': r3, 'ndcg@3': ndcg3, 'mrr': mrr
                }
            })
        
        # Step 6: Show detailed results
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS")
        print('='*80)
        
        for result in all_results:
            print(f"\n{result['system']} System:")
            print(f"  Retrieved docs: {result['retrieved_docs']}")
            print(f"  Relevant docs found: {result['relevant_found']}")
            print(f"  Success: {'' if result['relevant_found'] else ''}")
            
            if result['relevant_found']:
                print(f"  Found relevant doc(s): {result['relevant_found']}")
        
        # Return the dataset and data for further use
        return dataset, first_query_data
        
    except Exception as e:
        print(f"Error in evaluate_with_first_query: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# Update the test_first_query_integration function with better error handling:

def test_first_query_integration():
    """
    Integration test that combines first query evaluation with your existing systems.
    """
    
    print("=== Testing First Query Integration ===")
    
    try:
        # Get the first query dataset
        result = evaluate_with_first_query()
        
        if result is None or result == (None, None):
            print("Failed to get first query data. Using fallback test...")
            use_fallback_test()
            return
        
        dataset, first_query_data = result
        
        if dataset is None or first_query_data is None:
            print("Failed to get first query data. Using fallback test...")
            use_fallback_test()
            return
        
        # Test with your existing MarcoTopKEvaluator
        print(f"\n{'='*80}")
        print("USING YOUR EXISTING EVALUATOR")
        print('='*80)
        
        evaluator = MarcoTopKEvaluator(dataset, k_values=[1, 3, 5])
        
        # Create a test system that returns the relevant document first (perfect retrieval)
        class PerfectTestSystem:
            def __init__(self, relevant_doc_id, all_doc_ids):
                self.relevant_doc_id = relevant_doc_id
                self.all_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id != relevant_doc_id]
            
            def retrieve(self, query: str, top_k: int) -> List[str]:
                # Return relevant doc first, then others
                results = [self.relevant_doc_id] + self.all_doc_ids[:top_k-1]
                return results[:top_k]
        
        perfect_system = PerfectTestSystem(
            first_query_data['relevant_doc_id'],
            dataset.docs['id'].tolist()
        )
        
        # Evaluate with your existing system
        results = evaluator.evaluate_system_top_k(perfect_system, max_queries=1, verbose=True)
        
        print(f"\nPerfect System Results (should have high scores):")
        for metric, value in results['aggregate_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        if results['aggregate_metrics'].get('precision@1', 0) > 0:
            print(f"\nSUCCESS! Your evaluation system is working correctly with real MS MARCO data!")
        else:
            print(f"\nSomething is still not working correctly.")
            
    except Exception as e:
        print(f"Error in test_first_query_integration: {e}")
        import traceback
        traceback.print_exc()
        print("Using fallback test...")
        use_fallback_test()


def use_fallback_test():
    """Fallback test using simple test data."""
    
    print("=== Using Simple Test Data (No Download Required) ===")
    
    # Create simple test data
    docs_data = [
        {'id': 'doc_1', 'content': 'Paula Deen is a famous southern chef known for her cooking shows.'},
        {'id': 'doc_2', 'content': 'Earl W. Hiers Jr., also known as Bubba, is Paula Deen\'s brother.'},
        {'id': 'doc_3', 'content': 'Southern cooking involves traditional recipes from the American South.'},
        {'id': 'doc_4', 'content': 'Television cooking shows feature celebrity chefs and their recipes.'},
        {'id': 'doc_5', 'content': 'Restaurant business requires good management and quality food.'},
    ]
    
    queries_data = [
        {'id': 'q_1', 'text': 'what is paula deen\'s brother'}
    ]
    
    qrels_data = [
        {'query_id': 'q_1', 'doc_id': 'doc_2', 'relevance': 1}  # doc_2 is relevant
    ]
    
    # Create DataFrames
    docs_df = pd.DataFrame(docs_data)
    queries_df = pd.DataFrame(queries_data)
    qrels_df = pd.DataFrame(qrels_data)
    
    print(f"Created simple test data:")
    print(f"  Docs: {len(docs_df)}")
    print(f"  Queries: {len(queries_df)}")
    print(f"  Qrels: {len(qrels_df)}")
    
    # Create dataset (without name parameter)
    dataset = DataFrameDataset(docs_df, queries_df, qrels_df)
    dataset.name = "Simple Test Dataset"  # Set name after creation
    
    # Test with your systems
    from ms_macro_eval import MarcoRetrievalSystem
    
    # Create retrieval systems
    tfidf_system = MarcoRetrievalSystem(dataset.docs, method="tfidf")
    bm25_system = MarcoRetrievalSystem(dataset.docs, method="bm25")
    
    # Test retrieval
    query_text = 'what is paula deen\'s brother'
    relevant_docs = ['doc_2']
    relevance_scores = {'doc_2': 1.0}
    
    print(f"\nTesting retrieval systems:")
    print(f"Query: {query_text}")
    print(f"Expected relevant doc: doc_2")
    print(f"Doc_2 content: {docs_data[1]['content']}")
    
    systems = [('TF-IDF', tfidf_system), ('BM25', bm25_system)]
    
    print(f"\n{'System':<10} {'Retrieved Docs':<30} {'P@1':<6} {'MRR':<6} {'Success':<10}")
    print("-" * 70)
    
    for system_name, system in systems:
        retrieved_docs = system.retrieve(query_text, top_k=5)
        p1 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 1)
        mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        
        relevant_found = [doc for doc in retrieved_docs if doc in relevant_docs]
        success = "YES" if relevant_found else " NO"
        
        print(f"{system_name:<10} {str(retrieved_docs[:3]):<30} {p1:<6.3f} {mrr:<6.3f} {success:<10}")
        
        if relevant_found:
            print(f"  → Found relevant doc: {relevant_found[0]}")
    
    return dataset, {
        'query_text': query_text,
        'relevant_docs': relevant_docs,
        'relevance_scores': relevance_scores
    }


if __name__ == "__main__":
    # Run the first query evaluation
    test_first_query_integration()
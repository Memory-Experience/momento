def test_eval_dt():
    try:
        # Example usage of DataFrameDataset with ir_datasets
        print("Loading MS MARCO small dev dataset...")
        raw_dataset = ir_datasets.load("msmarco-passage/dev/small")
        
        print("Converting documents to DataFrame...")
        docs_data = [{'id': doc.doc_id, 'content': doc.text} for doc in raw_dataset.docs_iter()]
        
        print("Converting queries to DataFrame...")
        queries_data = [{'id': q.query_id, 'text': q.text} for q in raw_dataset.queries_iter()]
        
        print("Converting relevance judgments to DataFrame...")
        qrels_data = [{'query_id': qrel.query_id, 'doc_id': qrel.doc_id, 'relevance': qrel.relevance} 
                      for qrel in raw_dataset.qrels_iter()]
    except AttributeError as e:
        print(f"Error accessing dataset attributes: {e}")
        print("Available attributes:", dir(raw_dataset))
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Create DataFrames
    docs_df = pd.DataFrame(docs_data)
    queries_df = pd.DataFrame(queries_data)
    qrels_df = pd.DataFrame(qrels_data)
    
    # Create our DataFrameDataset
    dataset = DataFrameDataset(docs_df, queries_df, qrels_df)
    
    # Evaluate metrics for each query
    k_values = [1, 3, 5, 10]
    results = []
    
    # Get unique query IDs
    unique_queries = dataset.qrels['query_id'].unique()
    
    for query_id in unique_queries:
        relevant_docs = set(dataset.qrels[dataset.qrels['query_id'] == query_id]['doc_id'])
        
        # Simulate ranked results: relevant docs first, then fillers
        ranked_docs = list(relevant_docs) + [
            doc_id for doc_id in dataset.docs['id'] 
            if doc_id not in relevant_docs
        ][:max(k_values)]
        
        metrics = {'query_id': query_id, 'num_relevant': len(relevant_docs)}
        for k in k_values:
            metrics.update({
                f'precision@{k}': precision_at_k(ranked_docs, relevant_docs, k),
                f'recall@{k}': recall_at_k(ranked_docs, relevant_docs, k),
                f'ndcg@{k}': ndcg_at_k(ranked_docs, relevant_docs, k)
            })
        
        metrics['mrr'] = reciprocal_rank(ranked_docs, relevant_docs)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Print average metrics
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for k in k_values:
        print(f"\nMetrics at k={k}:")
        print(f"Average Precision@{k}: {results_df[f'precision@{k}'].mean():.3f}")
        print(f"Average Recall@{k}: {results_df[f'recall@{k}'].mean():.3f}")
        print(f"Average NDCG@{k}: {results_df[f'ndcg@{k}'].mean():.3f}")
    
    print(f"\nMean Reciprocal Rank: {results_df['mrr'].mean():.3f}")
    
    # Print dataset stats
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total number of documents: {len(dataset.docs)}")
    print(f"Total number of queries: {len(dataset.queries)}")
    print(f"Total number of relevance judgments: {len(dataset.qrels)}")
    print(f"Average relevant documents per query: {results_df['num_relevant'].mean():.2f}")

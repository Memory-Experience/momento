"""
MS MARCO Framework Demo with Guaranteed Working Example

This demo shows the generic evaluation framework working by creating
a small, aligned dataset that demonstrates the evaluation metrics clearly.
"""

import pandas as pd
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DemoMSMarcoAdapter:
    """A demo adapter with small, aligned MS MARCO-style data."""
    
    def __init__(self):
        # Create small demo dataset with guaranteed alignment
        self.queries_df = pd.DataFrame([
            {'id': 'q1', 'text': 'What is artificial intelligence?'},
            {'id': 'q2', 'text': 'How does machine learning work?'},
            {'id': 'q3', 'text': 'What are neural networks?'},
            {'id': 'q4', 'text': 'Explain deep learning concepts'},
            {'id': 'q5', 'text': 'What is natural language processing?'}
        ])
        
        self.docs_df = pd.DataFrame([
            {'id': 'd1', 'content': 'Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans. AI research deals with the question of how to create computers that are capable of intelligent behavior.'},
            {'id': 'd2', 'content': 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions.'},
            {'id': 'd3', 'content': 'Neural networks are computing systems inspired by biological neural networks. They are composed of nodes (neurons) that process information using a connectionist approach to computation.'},
            {'id': 'd4', 'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.'},
            {'id': 'd5', 'content': 'Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.'},
            {'id': 'd6', 'content': 'Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.'},
            {'id': 'd7', 'content': 'Robotics is an interdisciplinary branch of engineering and science that includes mechanical, electrical, and computer engineering.'},
            {'id': 'd8', 'content': 'The history of computers dates back to ancient calculating devices, but modern computers were developed in the 20th century.'}
        ])
        
        # Create relevance judgments that align with queries and docs
        self.qrels_df = pd.DataFrame([
            {'query_id': 'q1', 'doc_id': 'd1', 'relevance': 3},  # AI query -> AI doc (perfect match)
            {'query_id': 'q1', 'doc_id': 'd2', 'relevance': 2},  # AI query -> ML doc (related)
            {'query_id': 'q2', 'doc_id': 'd2', 'relevance': 3},  # ML query -> ML doc (perfect match)
            {'query_id': 'q2', 'doc_id': 'd4', 'relevance': 2},  # ML query -> DL doc (related)
            {'query_id': 'q3', 'doc_id': 'd3', 'relevance': 3},  # NN query -> NN doc (perfect match)
            {'query_id': 'q3', 'doc_id': 'd4', 'relevance': 2},  # NN query -> DL doc (related)
            {'query_id': 'q4', 'doc_id': 'd4', 'relevance': 3},  # DL query -> DL doc (perfect match)
            {'query_id': 'q4', 'doc_id': 'd3', 'relevance': 2},  # DL query -> NN doc (related)
            {'query_id': 'q5', 'doc_id': 'd5', 'relevance': 3},  # NLP query -> NLP doc (perfect match)
            {'query_id': 'q5', 'doc_id': 'd1', 'relevance': 1},  # NLP query -> AI doc (somewhat related)
        ])
        
        logging.info(" Demo MS MARCO adapter created with aligned data")
    
    def get_queries(self):
        return self.queries_df
    
    def get_documents(self):
        return self.docs_df
    
    def get_relevance_judgments(self):
        return self.qrels_df
    
    def get_name(self):
        return "Demo MS MARCO (AI/ML dataset)"

class IntelligentMockRAG:
    """A mock RAG that simulates different performance levels."""
    
    def __init__(self, performance_level="medium"):
        self.performance_level = performance_level
        logging.info(f" IntelligentMockRAG initialized (performance: {performance_level})")
    
    def search(self, query: str, top_k: int = 10):
        """Mock search with different performance patterns."""
        import random
        
        # Map queries to their most relevant documents (simulating a real system)
        query_to_docs = {
            'What is artificial intelligence?': ['d1', 'd2', 'd5'],  # AI, ML, NLP
            'How does machine learning work?': ['d2', 'd4', 'd3'],  # ML, DL, NN
            'What are neural networks?': ['d3', 'd4', 'd2'],       # NN, DL, ML
            'Explain deep learning concepts': ['d4', 'd3', 'd2'],   # DL, NN, ML
            'What is natural language processing?': ['d5', 'd1']   # NLP, AI
        }
        
        if query in query_to_docs:
            relevant_docs = query_to_docs[query]
            
            if self.performance_level == "perfect":
                # Always return relevant docs first
                results = relevant_docs[:top_k]
            elif self.performance_level == "good":
                # 80% chance to put relevant doc first
                results = []
                if random.random() < 0.8 and relevant_docs:
                    results.append(relevant_docs[0])
                
                # Add some other relevant docs
                for doc in relevant_docs[1:]:
                    if random.random() < 0.6:
                        results.append(doc)
                
                # Fill with random docs
                all_docs = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                remaining = [d for d in all_docs if d not in results]
                while len(results) < top_k and remaining:
                    results.append(remaining.pop(random.randint(0, len(remaining)-1)))
                    
            elif self.performance_level == "medium":
                # 50% chance to include relevant docs
                results = []
                for doc in relevant_docs:
                    if random.random() < 0.5:
                        results.append(doc)
                
                # Fill with random docs
                all_docs = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                remaining = [d for d in all_docs if d not in results]
                while len(results) < top_k and remaining:
                    results.append(remaining.pop(random.randint(0, len(remaining)-1)))
                    
            else:  # poor
                # Mostly random results
                all_docs = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                results = random.sample(all_docs, min(top_k, len(all_docs)))
            
            return results
        else:
            # Unknown query, return random results
            all_docs = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
            return random.sample(all_docs, min(top_k, len(all_docs)))

def main():
    """Run the MS MARCO working example."""
    print(" MS MARCO EVALUATION FRAMEWORK - WORKING EXAMPLE")
    print("=" * 65)
    print("This demo uses a small AI/ML dataset to show how evaluation works")
    
    # Import the generic framework
    try:
        from generic_evaluation_framework import GenericEvaluationEngine
    except ImportError as e:
        print(f" Import failed: {e}")
        return
    
    # Step 1: Create demo dataset
    print("\n STEP 1: Creating Demo MS MARCO-style Dataset")
    print("-" * 50)
    
    dataset = DemoMSMarcoAdapter()
    
    print(f" Dataset created: {dataset.get_name()}")
    print(f"    Queries: {len(dataset.get_queries())}")
    print(f"    Documents: {len(dataset.get_documents())}")
    print(f"    Relevance judgments: {len(dataset.get_relevance_judgments())}")
    
    # Show sample data
    print(f"\n Sample Query-Document Pairs:")
    queries = dataset.get_queries()
    docs = dataset.get_documents()
    qrels = dataset.get_relevance_judgments()
    
    for i in range(min(3, len(queries))):
        query = queries.iloc[i]
        relevant_qrels = qrels[qrels['query_id'] == query['id']]
        
        print(f"\n   Query: '{query['text']}'")
        for _, qrel in relevant_qrels.iterrows():
            doc = docs[docs['id'] == qrel['doc_id']].iloc[0]
            print(f"   → Relevant Doc (score {qrel['relevance']}): '{doc['content'][:80]}...'")
    
    # Step 2: Test different RAG performance levels
    performance_levels = ["perfect", "good", "medium", "poor"]
    
    for performance in performance_levels:
        print(f"\n STEP 2: Testing {performance.upper()} RAG Performance")
        print("-" * 50)
        
        rag_system = IntelligentMockRAG(performance_level=performance)
        evaluation_engine = GenericEvaluationEngine(dataset, rag_system)
        
        results = evaluation_engine.evaluate(
            sample_size=5,  # All our queries
            k_values=[1, 3, 5]
        )
        
        print(f"\n Results for {performance.upper()} RAG:")
        print(f"   Precision@1: {results['avg_metrics']['precision@1']:.2f} (How often top result is relevant)")
        print(f"   Recall@3: {results['avg_metrics']['recall@3']:.2f} (How many relevant docs found in top 3)")
        print(f"   MRR: {results['avg_metrics']['mrr']:.3f} (How quickly users find relevant results)")
        
        if performance == "perfect":
            print("    Perfect system: Always returns most relevant documents first")
        elif performance == "good":
            print("   Good system: Usually finds relevant docs, good user experience")
        elif performance == "medium":
            print("   Medium system: Sometimes finds relevant docs, moderate success")
        else:
            print("    Poor system: Rarely finds relevant docs, poor user experience")
    
    # Step 3: Detailed example with good performance
    print(f"\n STEP 3: Detailed Analysis with GOOD RAG System")
    print("-" * 55)
    
    rag_system = IntelligentMockRAG(performance_level="good")
    evaluation_engine = GenericEvaluationEngine(dataset, rag_system)
    
    results = evaluation_engine.evaluate(sample_size=5, k_values=[1, 3, 5])
    
    print(f"\n Detailed Metrics:")
    for metric, value in results['avg_metrics'].items():
        print(f"   {metric}: {value:.3f}")
    
    print(f"\n Per-Query Analysis:")
    for query_result in results['query_results']:
        print(f"\n   Query: '{query_result['query_text']}'")
        print(f"   → Available relevant docs: {query_result['relevant_count']}")
        print(f"   → Precision@1: {query_result['metrics']['precision@1']:.2f}")
        print(f"   → Recall@3: {query_result['metrics']['recall@3']:.2f}")
        
        if query_result['metrics']['precision@1'] == 1.0:
            print("    Success! User gets exactly what they want immediately")
        elif query_result['metrics']['recall@3'] > 0.5:
            print("   Good! User can find relevant info with some browsing")
        else:
            print("    Poor! User might struggle to find what they need")
    
    print(f"\n STEP 4: What This Means for Personal Memory Systems")
    print("-" * 65)
    
    print("This evaluation framework demonstrates:")
    print(" How to measure retrieval quality with standard IR metrics")
    print(" How different system performance affects user experience")
    print(" How to identify specific queries where systems fail")
    print(" How to track improvements over time")
    print()
    print("For personal memory use cases:")
    print("Same framework, same metrics, same analysis")
    print("Just replace AI/ML docs → User's personal memories")
    print("Replace general queries → User's personal questions")
    print("Replace mock RAG → Vector database search")
    print()
    print(" The metrics directly translate to user satisfaction:")
    print("   • High Precision@1 → Users get what they want immediately")
    print("   • High Recall@3 → Users find all their relevant memories")
    print("   • High MRR → Users quickly find what they're looking for")
    
    print(f"\n MS MARCO FRAMEWORK DEMONSTRATION COMPLETE!")
    print(" Ready to evaluate personal memory systems!")

if __name__ == "__main__":
    main()

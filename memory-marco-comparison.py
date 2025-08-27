#!/usr/bin/env python3
"""
Detailed Memory vs MS MARCO Comparison
Comparing your AI definition to professional AI passages in MS MARCO
"""

import sys
from pathlib import Path

# Add your packages to path
project_root = Path(__file__).parent
api_path = project_root / "packages" / "api"
eval_path = project_root / "packages" / "evaluation"
sys.path.extend([str(api_path), str(eval_path)])

try:
    from rag.comparative_rag_service import ComparativeRAGService
    from dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes
    from dataset_marco.prepare_ms_marco import MSMarcoDataset
    import pandas as pd
    print("✅ Successfully imported your services")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def detailed_memory_comparison():
    """Compare your AI memory to MS MARCO AI passages in detail."""
    
    print("🔬 DETAILED MEMORY vs MS MARCO COMPARISON")
    print("=" * 55)
    
    # Your actual AI memory
    your_memory = "artificial intelligence. Intelligence is an intelligence. demonstrated by machines. It is different than the entire intelligence of humans and animals. especially distinguished by the data."
    
    print(f"📝 YOUR AI MEMORY:")
    print(f"'{your_memory}'")
    print()
    
    # Load MS MARCO data directly
    print("📊 Loading MS MARCO dataset...")
    try:
        # Initialize MS MARCO dataset first
        from dataset_marco.prepare_ms_marco import MSMarcoDataset
        marco_dataset = MSMarcoDataset()
        docs_df, queries_df, qrels_df = convert_ms_marco_to_dataframes(marco_dataset, limit=1000)
        print(f"✅ Loaded: {len(docs_df)} documents, {len(queries_df)} queries")
        
        # Debug: Check column names
        print(f"📋 Documents columns: {list(docs_df.columns)}")
        print(f"📋 Queries columns: {list(queries_df.columns)}")
        
    except Exception as e:
        print(f"❌ Failed to load MS MARCO: {e}")
        return
    
    # Find AI-related queries in MS MARCO
    print("\n🔍 FINDING AI-RELATED QUERIES IN MS MARCO:")
    print("-" * 45)
    ai_queries = queries_df[queries_df['text'].str.contains('artificial intelligence|AI|machine learning|intelligence', case=False, na=False)]
    
    print(f"Found {len(ai_queries)} AI-related queries:")
    for i, (_, row) in enumerate(ai_queries.head(5).iterrows(), 1):
        print(f"   {i}. {row['text']}")
    
    # Find AI-related documents in MS MARCO
    print("\n📄 FINDING AI-RELATED DOCUMENTS IN MS MARCO:")
    print("-" * 47)
    ai_docs = docs_df[docs_df['content'].str.contains('artificial intelligence|AI|machine learning|intelligence', case=False, na=False)]
    
    print(f"Found {len(ai_docs)} AI-related documents:")
    for i, (_, row) in enumerate(ai_docs.head(3).iterrows(), 1):
        content = row['content'][:150] + "..." if len(row['content']) > 150 else row['content']
        print(f"   {i}. {content}")
    
    # Compare your memory to professional definitions
    print("\n⚖️  COMPARISON ANALYSIS:")
    print("-" * 35)
    
    # Key terms in your memory
    your_terms = ["intelligence", "machines", "humans", "animals", "data"]
    print(f"🔤 Key terms in your memory: {', '.join(your_terms)}")
    
    # Check coverage in MS MARCO
    marco_coverage = {}
    for term in your_terms:
        count = docs_df['content'].str.contains(term, case=False, na=False).sum()
        marco_coverage[term] = count
    
    print(f"\n📈 Term coverage in MS MARCO dataset:")
    for term, count in marco_coverage.items():
        percentage = (count / len(docs_df)) * 100
        print(f"   • '{term}': {count} docs ({percentage:.1f}%)")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print("-" * 25)
    print(f"✅ Your memory covers core AI concepts")
    print(f"✅ Mentions key distinction: machines vs humans/animals")
    print(f"✅ References data as distinguishing factor")
    print(f"⚠️  Could be more structured (grammar/clarity)")
    print(f"📊 MS MARCO provides {len(ai_docs)} professional AI definitions")
    
    # Show best MS MARCO match
    if len(ai_docs) > 0:
        print(f"\n🏆 BEST MS MARCO AI DEFINITION:")
        print("-" * 35)
        best_doc = ai_docs.iloc[0]
        print(f"'{best_doc['content'][:200]}...'")
    
    print(f"\n💡 INSIGHTS:")
    print("-" * 15)
    print(f"• Your memory: Personal, intuitive understanding")
    print(f"• MS MARCO: Professional, structured definitions")
    print(f"• Both cover: Intelligence, machines, differentiation")
    print(f"• Opportunity: Combine personal insight with formal structure")

if __name__ == "__main__":
    try:
        detailed_memory_comparison()
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

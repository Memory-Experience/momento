#!/usr/bin/env python3
"""
Quick debug to see MS MARCO DataFrame columns
"""

import sys
from pathlib import Path

# Add the evaluation package to Python path
eval_path = Path(__file__).parent / "packages" / "evaluation"
sys.path.append(str(eval_path))

from dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes
from dataset_marco.prepare_ms_marco import MSMarcoDataset

try:
    print("🔍 Loading MS MARCO dataset...")
    marco_dataset = MSMarcoDataset("msmarco-passage/dev/small")
    queries, docs, qrels = convert_ms_marco_to_dataframes(marco_dataset, limit=10)
    
    print(f"\n📊 DataFrame Structures:")
    print(f"Documents columns: {list(docs.columns)}")
    print(f"Queries columns: {list(queries.columns)}")
    print(f"QRELs columns: {list(qrels.columns)}")
    
    print(f"\n📄 Sample Document:")
    print(docs.iloc[0])
    
    print(f"\n❓ Sample Query:")
    print(queries.iloc[0])
    
    print(f"\n🎯 Sample QREL:")
    print(qrels.iloc[0])
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

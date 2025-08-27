# Generic RAG Evaluation Framework

**A unified framework to evaluate RAG systems against multiple datasets including MS MARCO and personal memories.**

## What This Framework Provides

### **Generic Evaluation Engine**
- Works with **any dataset** (MS MARCO, personal memories, custom datasets)
- **Standardized metrics**: Precision@k, Recall@k, NDCG@k, MRR
- **Easy to extend** with new datasets and retrieval systems
- **Production-ready** with comprehensive error handling

### **Multiple Dataset Support**
- **MS MARCO**: Industry-standard IR benchmark
- **Personal Memory**: Evaluate personal memory retrieval systems  
- **Custom**: Easy to add new datasets via adapters

### **RAG System Integration**
- **Existing RAG systems**: Simple adapter to connect current systems
- **Mock systems**: For testing and development
- **Comparative analysis**: Compare different approaches

## Framework Structure

```
packages/evaluation/
├── generic_evaluation_framework.py     # Core evaluation engine
├── complete_evaluation_demo.py         # MS MARCO evaluation demo
├── personal_memory_evaluation_demo.py  # Personal memory demo
│
├── dataset_adapters/
│   ├── marco_adapter.py               # MS MARCO dataset adapter  
│   ├── personal_memory_adapter.py     # Personal memory adapter
│   └── rag_adapter.py                 # RAG system adapters
│
├── dataset_marco/                     # MS MARCO specific code
└── sample_personal_memory_dataset/    # Synthetic personal data
```

## Use Case: Personal Memory Evaluation

### **Scenario:**
1. **User Query**: "Tell me about my vacation in France"
2. **Memory Database**: User's recorded memories stored as vectors
3. **Retrieval**: System finds top-k relevant memories
4. **Gold Standard**: Which memories SHOULD be retrieved
5. **Evaluation**: How well did the system perform?

### **Key Metrics:**
- **Precision@1**: How often is the TOP result actually relevant?
- **Recall@5**: How many relevant memories are found in top 5?
- **MRR**: How quickly does the user find what they want?

## Quick Start

### **1. Evaluate with MS MARCO (Generic IR Benchmark)**
```bash
cd packages/evaluation
python complete_evaluation_demo.py
```

### **2. Evaluate Personal Memory Retrieval**
```bash
cd packages/evaluation  
python personal_memory_evaluation_demo.py
```

### **3. Integrate Your RAG System**
```python
from generic_evaluation_framework import GenericEvaluationEngine
from dataset_adapters.marco_adapter import create_ms_marco_adapter
from dataset_adapters.rag_adapter import create_rag_adapter

# Load dataset (MS MARCO or personal memories)
dataset = create_ms_marco_adapter(limit=1000)

# Connect RAG system
rag_system = create_rag_adapter("comparative")

# Run evaluation
engine = GenericEvaluationEngine(dataset, rag_system)
results = engine.evaluate(sample_size=100)
engine.print_results(results)
```

## Adding New Datasets

To evaluate against custom data, create a new adapter:

```python
from generic_evaluation_framework import DatasetInterface
import pandas as pd

class MyDatasetAdapter(DatasetInterface):
    def get_queries(self) -> pd.DataFrame:
        # Return DataFrame with columns: ['id', 'text']
        return your_queries_df
    
    def get_documents(self) -> pd.DataFrame:
        # Return DataFrame with columns: ['id', 'content'] 
        return your_documents_df
    
    def get_relevance_judgments(self) -> pd.DataFrame:
        # Return DataFrame with columns: ['query_id', 'doc_id', 'relevance']
        return your_relevance_df
```

## Adding New RAG Systems

To evaluate RAG systems, create an adapter:

```python
from generic_evaluation_framework import RetrievalSystemInterface

class MyRAGAdapter(RetrievalSystemInterface):
    def search(self, query: str, top_k: int = 10) -> List[str]:
        # Retrieval logic here
        results = rag_system.search(query, top_k)
        return [result.id for result in results]
```

## Understanding Results

### **Precision@k**: Quality of retrieved results
- `1.0` = Perfect! All retrieved docs are relevant
- `0.5` = Half of retrieved docs are relevant  
- `0.0` = No relevant docs retrieved

### **Recall@k**: Coverage of relevant documents
- `1.0` = Perfect! Found all relevant docs
- `0.5` = Found half of all relevant docs
- `0.0` = Didn't find any relevant docs

### **NDCG@k**: Ranking quality with graded relevance
- `1.0` = Perfect ranking (most relevant docs first)
- `0.5` = Decent ranking but room for improvement
- `0.0` = Poor ranking

### **MRR**: User experience (how fast they find relevant docs)
- `1.0` = First result is always relevant
- `0.5` = Relevant result typically at position 2
- `0.33` = Relevant result typically at position 3

## Real-World Application

### **For Personal Memory Systems:**
1. **Export** user memories from your vector database
2. **Create** realistic user queries (or use actual user queries)
3. **Label** which memories are relevant (gold standard)
4. **Connect** actual RAG service via adapter
5. **Evaluate** and iterate on improvements

### **Improvement Areas Based on Results:**
- **Low Precision**: Improve embedding models, tune similarity thresholds
- **Low Recall**: Expand search, include metadata, improve chunking
- **Low NDCG**: Better ranking algorithms, personalization
- **Low MRR**: Optimize for user intent, context understanding

## Extensions

This framework is designed to grow with various needs:

### **Dataset Extensions:**
- Customer support knowledge bases
- Medical records retrieval
- Legal document search
- Scientific paper collections

### **Metric Extensions:**
- Custom domain-specific metrics
- User satisfaction scores
- Latency measurements
- Cost analysis

### **System Extensions:**
- A/B testing different RAG approaches
- Multi-modal retrieval (text + images)
- Federated search across multiple sources
- Real-time evaluation during production

## Production Usage

### **Continuous Evaluation:**
```python
# Set up automated evaluation pipeline
def evaluate_production_system():
    dataset = load_latest_user_queries_and_feedback()
    rag_system = ProductionRAGAdapter()
    
    engine = GenericEvaluationEngine(dataset, rag_system)
    results = engine.evaluate()
    
    # Alert if metrics drop below thresholds
    if results['avg_metrics']['precision@1'] < 0.8:
        send_alert("RAG performance degraded!")
    
    return results
```

### **A/B Testing:**
```python
# Compare different RAG approaches
def compare_rag_systems():
    dataset = create_ms_marco_adapter()
    
    old_system = create_rag_adapter("simple")
    new_system = create_rag_adapter("enhanced")
    
    old_results = GenericEvaluationEngine(dataset, old_system).evaluate()
    new_results = GenericEvaluationEngine(dataset, new_system).evaluate()
    
    print(f"Old P@1: {old_results['avg_metrics']['precision@1']:.3f}")
    print(f"New P@1: {new_results['avg_metrics']['precision@1']:.3f}")
```

---

**This framework provides tools to systematically improve RAG systems by providing clear, actionable metrics that directly relate to user experience and satisfaction.**

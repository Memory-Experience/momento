"""
Personal Memory Dataset Adapter for Generic Evaluation Framework

This adapter shows how to extend the generic evaluation framework to work with
personal memory datasets. This is a template that can customize for
specific memory storage format.

Example Use Case:
- User asks: "Tell me about my vacation in France"
- Memory DB: Contains user's recorded memories as vectors
- Gold Standard: User manually marks which memories are relevant
- Evaluation: How well does retrieval find the RIGHT personal memories?
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Import generic framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from generic_evaluation_framework import DatasetInterface

class PersonalMemoryDatasetAdapter(DatasetInterface):
    """
    Adapter for personal memory datasets.
    
    This adapter expects data in the following format:
    - memories.csv: id, content, timestamp, metadata
    - queries.csv: id, text, user_intent  
    - relevance.csv: query_id, memory_id, relevance (0-3 scale)
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize personal memory dataset adapter.
        
        Args:
            data_dir: Directory containing the memory dataset files
        """
        self.data_dir = Path(data_dir)
        
        # Load data files
        try:
            self.memories_df = pd.read_csv(self.data_dir / "memories.csv")
            self.queries_df = pd.read_csv(self.data_dir / "queries.csv")
            self.relevance_df = pd.read_csv(self.data_dir / "relevance.csv")
            
            logging.info(f" Personal memory dataset loaded from {data_dir}")
            logging.info(f"    Memories: {len(self.memories_df)}")
            logging.info(f"    Queries: {len(self.queries_df)}")
            logging.info(f"    Relevance judgments: {len(self.relevance_df)}")
            
        except FileNotFoundError as e:
            logging.error(f" Dataset file not found: {e}")
            raise
        except Exception as e:
            logging.error(f" Failed to load personal memory dataset: {e}")
            raise
    
    def get_queries(self) -> pd.DataFrame:
        """Return queries DataFrame with columns: ['id', 'text']"""
        # Ensure standard column names
        result = self.queries_df.copy()
        if 'query_text' in result.columns:
            result = result.rename(columns={'query_text': 'text'})
        return result[['id', 'text']]
    
    def get_documents(self) -> pd.DataFrame:
        """Return documents (memories) DataFrame with columns: ['id', 'content']"""
        # Ensure standard column names
        result = self.memories_df.copy()
        if 'memory_content' in result.columns:
            result = result.rename(columns={'memory_content': 'content'})
        elif 'text' in result.columns:
            result = result.rename(columns={'text': 'content'})
        return result[['id', 'content']]
    
    def get_relevance_judgments(self) -> pd.DataFrame:
        """Return relevance judgments DataFrame with columns: ['query_id', 'doc_id', 'relevance']"""
        # Ensure standard column names
        result = self.relevance_df.copy()
        if 'memory_id' in result.columns:
            result = result.rename(columns={'memory_id': 'doc_id'})
        return result[['query_id', 'doc_id', 'relevance']]
    
    def get_name(self) -> str:
        """Return dataset name for logging/reporting"""
        return f"Personal Memory Dataset ({self.data_dir.name})"

class SyntheticPersonalMemoryDataset:
    """
    Creates synthetic personal memory datasets for testing.
    Useful for development and demonstration purposes.
    """
    
    @staticmethod
    def create_sample_dataset(output_dir: Path, num_memories: int = 50, num_queries: int = 10):
        """
        Create a sample personal memory dataset.
        
        Args:
            output_dir: Directory to save the dataset files
            num_memories: Number of synthetic memories to create
            num_queries: Number of synthetic queries to create
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Sample memory topics
        memory_templates = [
            "Vacation in {location} was amazing. We visited {activity} and enjoyed {food}.",
            "Work meeting about {topic}. Discussed {details} with the team.",
            "Family dinner at {restaurant}. Had {dish} and talked about {conversation}.",
            "Book reading session: {book_title}. Key insights: {insight}.",
            "Exercise routine: {exercise_type} for {duration}. Felt {feeling}.",
            "Learning about {subject}. Important concepts: {concepts}.",
            "Social event with {people}. Activities included {activities}.",
            "Travel to {destination}. Transportation: {transport}. Highlights: {highlights}.",
            "Cooking {recipe}. Ingredients: {ingredients}. Result: {outcome}.",
            "Reflection on {topic}. Thoughts: {thoughts}."
        ]
        
        # Generate synthetic memories
        memories_data = []
        for i in range(num_memories):
            template = memory_templates[i % len(memory_templates)]
            content = template.format(
                location=f"Location{i%10}",
                activity=f"Activity{i%8}",
                food=f"Food{i%6}",
                topic=f"Topic{i%12}",
                details=f"Details{i%15}",
                restaurant=f"Restaurant{i%7}",
                dish=f"Dish{i%9}",
                conversation=f"Conversation{i%11}",
                book_title=f"Book{i%13}",
                insight=f"Insight{i%14}",
                exercise_type=f"Exercise{i%5}",
                duration=f"{30+i%60} minutes",
                feeling=f"Feeling{i%8}",
                subject=f"Subject{i%10}",
                concepts=f"Concepts{i%12}",
                people=f"Friends{i%6}",
                activities=f"Activities{i%9}",
                destination=f"Destination{i%8}",
                transport=f"Transport{i%4}",
                highlights=f"Highlights{i%10}",
                recipe=f"Recipe{i%7}",
                ingredients=f"Ingredients{i%11}",
                outcome=f"Outcome{i%6}",
                thoughts=f"Thoughts{i%13}"
            )
            
            memories_data.append({
                'id': f"memory_{i}",
                'content': content,
                'timestamp': f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}",
                'metadata': f"category_{i%5}"
            })
        
        memories_df = pd.DataFrame(memories_data)
        memories_df.to_csv(output_dir / "memories.csv", index=False)
        
        # Generate synthetic queries
        query_templates = [
            "Tell me about my vacation in Location{loc}",
            "What did I learn about Subject{subj}?",
            "Recall my work meeting about Topic{topic}",
            "What book did I read recently?",
            "What exercise did I do?",
            "Where did I travel to?",
            "What did I cook?",
            "Who did I meet socially?",
            "What insights did I gain?",
            "What conversations did I have?"
        ]
        
        queries_data = []
        for i in range(num_queries):
            template = query_templates[i % len(query_templates)]
            text = template.format(
                loc=i%10,
                subj=i%10,
                topic=i%12
            )
            
            queries_data.append({
                'id': f"query_{i}",
                'text': text,
                'user_intent': f"recall_memory"
            })
        
        queries_df = pd.DataFrame(queries_data)
        queries_df.to_csv(output_dir / "queries.csv", index=False)
        
        # Generate relevance judgments (synthetic ground truth)
        relevance_data = []
        for query_idx in range(num_queries):
            query_id = f"query_{query_idx}"
            
            # Create some relevant memories for each query
            relevant_count = min(3, num_memories)  # Up to 3 relevant memories per query
            for rel_idx in range(relevant_count):
                memory_idx = (query_idx * 5 + rel_idx) % num_memories  # Distribute relevance
                memory_id = f"memory_{memory_idx}"
                
                # Assign relevance score (3=highly relevant, 2=relevant, 1=somewhat relevant)
                relevance_score = 3 if rel_idx == 0 else (2 if rel_idx == 1 else 1)
                
                relevance_data.append({
                    'query_id': query_id,
                    'memory_id': memory_id,
                    'relevance': relevance_score
                })
        
        relevance_df = pd.DataFrame(relevance_data)
        relevance_df.to_csv(output_dir / "relevance.csv", index=False)
        
        logging.info(f" Synthetic personal memory dataset created at {output_dir}")
        logging.info(f"    Memories: {len(memories_df)}")
        logging.info(f"    Queries: {len(queries_df)}")
        logging.info(f"    Relevance judgments: {len(relevance_df)}")
        
        return output_dir

def create_personal_memory_adapter(data_dir: Path) -> Optional[PersonalMemoryDatasetAdapter]:
    """
    Factory function to create personal memory adapter with error handling.
    
    Args:
        data_dir: Directory containing the memory dataset files
        
    Returns:
        PersonalMemoryDatasetAdapter instance or None if failed
    """
    try:
        return PersonalMemoryDatasetAdapter(data_dir)
    except Exception as e:
        logging.error(f" Failed to create personal memory adapter: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Create synthetic dataset for testing
    sample_dir = Path("./sample_personal_memory_dataset")
    
    print(" Creating synthetic personal memory dataset...")
    SyntheticPersonalMemoryDataset.create_sample_dataset(
        output_dir=sample_dir,
        num_memories=30,
        num_queries=8
    )
    
    # Test the adapter
    adapter = create_personal_memory_adapter(sample_dir)
    
    if adapter:
        print(f" Adapter created: {adapter.get_name()}")
        print(f" Queries: {len(adapter.get_queries())}")
        print(f" Memories: {len(adapter.get_documents())}")
        print(f" Relevance judgments: {len(adapter.get_relevance_judgments())}")
        
        # Show sample data
        print(f"\n Sample Query:")
        sample_query = adapter.get_queries().iloc[0]
        print(f"  ID: {sample_query['id']}")
        print(f"  Text: {sample_query['text']}")
        
        print(f"\n Sample Memory:")
        sample_memory = adapter.get_documents().iloc[0]
        print(f"  ID: {sample_memory['id']}")
        print(f"  Content: {sample_memory['content'][:100]}...")
    else:
        print(" Failed to create adapter")

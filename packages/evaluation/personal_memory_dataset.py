"""
Personal Memory Evaluation Dataset

This creates evaluation data for personal memory retrieval systems.
Instead of MS MARCO's general queries, we evaluate personal memory queries.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

class PersonalMemoryDataset:
    """Dataset for evaluating personal memory retrieval systems."""
    
    def __init__(self, recordings_dir: str):
        self.recordings_dir = Path(recordings_dir)
        self.memories = []
        self.queries = []
        self.qrels = []  # Query-memory relevance judgments
        
    def load_user_recordings(self, user_id: str = "default") -> List[Dict]:
        """Load user's recorded memories from recordings directory."""
        memories = []
        
        # Load all JSON files (recordings metadata)
        for json_file in self.recordings_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    recording_data = json.load(f)
                
                # Load corresponding transcription
                txt_file = json_file.with_suffix('.txt')
                transcription = ""
                if txt_file.exists():
                    with open(txt_file, 'r') as f:
                        transcription = f.read().strip()
                
                if transcription:
                    memory = {
                        'memory_id': json_file.stem,
                        'user_id': user_id,
                        'content': transcription,
                        'timestamp': recording_data.get('timestamp', datetime.now().isoformat()),
                        'metadata': recording_data
                    }
                    memories.append(memory)
                    
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
        
        print(f" Loaded {len(memories)} personal memories for user {user_id}")
        return memories
    
    def generate_personal_queries(self, memories: List[Dict]) -> List[Dict]:
        """Generate realistic personal memory queries based on actual memories."""
        queries = []
        
        # Extract topics/themes from memories for query generation
        memory_topics = self._extract_topics_from_memories(memories)
        
        # Generate different types of personal queries
        query_templates = [
            # Direct content queries
            ("What did I say about {topic}?", "content_specific"),
            ("Tell me about my {topic}", "topic_recall"),
            ("When did I mention {topic}?", "temporal_query"),
            
            # Contextual queries  
            ("What was I talking about on {date}?", "date_specific"),
            ("What were my thoughts about {topic}?", "opinion_recall"),
            ("Did I ever discuss {topic}?", "existence_query"),
            
            # Semantic queries
            ("What experiences did I share about {topic}?", "experience_recall"),
            ("How did I feel about {topic}?", "emotion_query"),
            ("What details did I mention about {topic}?", "detail_query")
        ]
        
        query_id = 0
        for topic_info in memory_topics:
            topic = topic_info['topic']
            related_memories = topic_info['memory_ids']
            
            for template, query_type in query_templates[:3]:  # Limit for demo
                query = {
                    'query_id': f"q_{query_id}",
                    'user_id': 'default',
                    'text': template.format(topic=topic),
                    'query_type': query_type,
                    'expected_memories': related_memories,
                    'topic': topic
                }
                queries.append(query)
                query_id += 1
        
        return queries
    
    def _extract_topics_from_memories(self, memories: List[Dict]) -> List[Dict]:
        """Extract topics/themes from memories for query generation."""
        topics = []
        
        # Simple keyword-based topic extraction
        topic_keywords = {
            'vacation': ['vacation', 'holiday', 'trip', 'travel', 'france', 'paris'],
            'work': ['work', 'job', 'office', 'meeting', 'project', 'boss'],
            'family': ['family', 'mother', 'father', 'sister', 'brother', 'parents'],
            'food': ['food', 'restaurant', 'dinner', 'lunch', 'cooking', 'meal'],
            'friends': ['friend', 'friends', 'party', 'social', 'hangout'],
            'health': ['doctor', 'health', 'exercise', 'gym', 'medical'],
            'hobbies': ['hobby', 'music', 'book', 'movie', 'sport', 'game']
        }
        
        for topic, keywords in topic_keywords.items():
            related_memories = []
            
            for memory in memories:
                content_lower = memory['content'].lower()
                if any(keyword in content_lower for keyword in keywords):
                    related_memories.append(memory['memory_id'])
            
            if related_memories:
                topics.append({
                    'topic': topic,
                    'memory_ids': related_memories,
                    'count': len(related_memories)
                })
        
        return topics
    
    def create_relevance_judgments(self, queries: List[Dict], memories: List[Dict]) -> List[Dict]:
        """Create query-memory relevance judgments (qrels)."""
        qrels = []
        
        memory_dict = {m['memory_id']: m for m in memories}
        
        for query in queries:
            query_id = query['query_id']
            query_text = query['text'].lower()
            query_topic = query.get('topic', '')
            
            # For each memory, determine relevance
            for memory in memories:
                memory_id = memory['memory_id']
                memory_content = memory['content'].lower()
                
                # Calculate relevance (0-3 scale like MS MARCO)
                relevance = self._calculate_memory_relevance(
                    query_text, query_topic, memory_content, query.get('expected_memories', [])
                )
                
                if relevance > 0:  # Only store relevant judgments
                    qrels.append({
                        'query_id': query_id,
                        'memory_id': memory_id,
                        'relevance': relevance,
                        'user_id': 'default'
                    })
        
        return qrels
    
    def _calculate_memory_relevance(self, query_text: str, query_topic: str, 
                                  memory_content: str, expected_memories: List[str]) -> int:
        """Calculate relevance score between query and memory."""
        
        # If memory is in expected memories (high relevance)
        memory_id = memory_content.split()[0] if memory_content else ""
        if any(mem_id in memory_content for mem_id in expected_memories):
            return 3  # Highly relevant
        
        # Topic-based relevance
        if query_topic and query_topic in memory_content:
            return 2  # Relevant
        
        # Keyword overlap relevance
        query_words = set(query_text.split())
        memory_words = set(memory_content.split())
        overlap = len(query_words.intersection(memory_words))
        
        if overlap >= 3:
            return 2  # Relevant
        elif overlap >= 1:
            return 1  # Partially relevant
        
        return 0  # Not relevant
    
    def create_evaluation_dataset(self, user_id: str = "default") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create complete evaluation dataset for personal memory retrieval."""
        
        print("🧠 Creating Personal Memory Evaluation Dataset")
        print("=" * 50)
        
        # Load user memories
        memories = self.load_user_recordings(user_id)
        
        # Generate personal queries
        queries = self.generate_personal_queries(memories)
        
        # Create relevance judgments
        qrels = self.create_relevance_judgments(queries, memories)
        
        # Convert to DataFrames
        memories_df = pd.DataFrame(memories)
        queries_df = pd.DataFrame(queries)
        qrels_df = pd.DataFrame(qrels)
        
        print(f" Dataset created:")
        print(f"   • Memories: {len(memories_df)}")
        print(f"   • Queries: {len(queries_df)}")
        print(f"   • Relevance judgments: {len(qrels_df)}")
        
        return memories_df, queries_df, qrels_df

    def save_dataset(self, memories_df: pd.DataFrame, queries_df: pd.DataFrame, 
                    qrels_df: pd.DataFrame, output_dir: str = "personal_memory_dataset"):
        """Save the evaluation dataset to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        memories_df.to_csv(output_path / "memories.csv", index=False)
        queries_df.to_csv(output_path / "queries.csv", index=False)
        qrels_df.to_csv(output_path / "qrels.csv", index=False)
        
        print(f"💾 Dataset saved to {output_path}/")


# Example usage
def create_personal_memory_dataset():
    """Example of creating personal memory evaluation dataset."""
    
    # Initialize dataset creator
    recordings_dir = Path(__file__).parent.parent.parent / "api" / "recordings"
    dataset_creator = PersonalMemoryDataset(str(recordings_dir))
    
    # Create dataset
    memories_df, queries_df, qrels_df = dataset_creator.create_evaluation_dataset()
    
    # Save dataset
    dataset_creator.save_dataset(memories_df, queries_df, qrels_df)
    
    # Show sample data
    print("\n SAMPLE DATA:")
    print("\nSample Queries:")
    print(queries_df[['query_id', 'text', 'topic']].head())
    
    print("\nSample Memories:")
    print(memories_df[['memory_id', 'content']].head())
    
    print("\nSample Relevance Judgments:")
    print(qrels_df.head())
    
    return memories_df, queries_df, qrels_df


if __name__ == "__main__":
    create_personal_memory_dataset()

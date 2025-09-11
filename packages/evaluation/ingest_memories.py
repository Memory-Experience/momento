import logging
import pickle

from tqdm import tqdm

from dataset.marco_dataset import MSMarcoDataset
from api.models.embedding.qwen3_embedding import Qwen3EmbeddingModel
from api.models.spacy_sentence_chunker import SpacySentenceChunker
from api.vector_store.repositories.qdrant_vector_store_repository import LocalFileQdrantVectorStoreRepository
from api.vector_store.repositories.vector_store_repository_interface import VectorStoreRepository
from api.vector_store.vector_store_service import VectorStoreService
from api.domain.memory_request import MemoryRequest

import gc
import sys
from pathlib import Path
import asyncio

# Add the root directory to sys.path
root_dir = Path(__file__).resolve().parents[2]  # Adjust based on your directory structure
sys.path.append(str(root_dir / "packages/api"))


LIMIT = 10_000


# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def process_documents(vector_store_service: VectorStoreService):
    dataset = MSMarcoDataset(limit=LIMIT)  # Limit to 100 items for faster evaluation
    logger.info(f"Loaded dataset: {dataset}")

    docs_df = dataset.docs[["id", "content"]]  # Keep only necessary columns
    total_docs = len(docs_df) if LIMIT is None else min(LIMIT, len(docs_df))

    doc_to_memory: dict[str, str] = {}
    memory_to_doc: dict[str, str] = {}

    for i, doc in enumerate(tqdm(docs_df.itertuples(index=False), total=total_docs)):
        current_passage = MemoryRequest.create(text=[str(doc[1])])

        await vector_store_service.index_memory(current_passage)
        
        dataset_doc_id = str(doc[0])
        saved_memory_id = current_passage.id

        doc_to_memory[dataset_doc_id] = saved_memory_id
        memory_to_doc[saved_memory_id] = dataset_doc_id

        gc.collect()

    with open("doc_to_memory.pkl", "wb") as f:
        pickle.dump(doc_to_memory, f)

    with open("memory_to_doc.pkl", "wb") as f:
        pickle.dump(memory_to_doc, f)

    logger.info("Saved doc_to_memory and memory_to_doc to pickle files.")

if __name__ == "__main__":
    embedding_model = Qwen3EmbeddingModel()
    text_chunker = SpacySentenceChunker()
    vector_store_repo: VectorStoreRepository = LocalFileQdrantVectorStoreRepository(
        embedding_model, text_chunker,
        database_path="qdrant_ms_marco_small"
    )
    vector_store_service = VectorStoreService(vector_store_repo)
    
    
    asyncio.run(process_documents(vector_store_service))
    
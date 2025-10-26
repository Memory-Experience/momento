import logging
import os
import pickle

from tqdm import tqdm

from dataset.dataset import DataFrameDataset
from api.vector_store.vector_store_service import VectorStoreService
from api.domain.memory_request import MemoryRequest
from api.vector_store.repositories.qdrant_vector_store_repository import (
    ServerQdrantVectorStoreRepository,
)
from api.vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)
from api.models.embedding.embedding_model_interface import EmbeddingModel

import gc

from baseline.dummy_text_chunker import DummyTextChunker


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load datasets and create filled vector store services with mappings.

    This class handles dataset ingestion into vector stores, including
    persistence of document-to-memory mappings for evaluation purposes.
    """

    def _create_vector_store_service(
        embedding: EmbeddingModel, dataset_folder: str,
    ) -> VectorStoreService:
        """
        Create a vector store service with local file-based Qdrant backend.

        Args:
            embedding (EmbeddingModel): Embedding model for vector storage
            dataset_folder (str): Path to folder for vector store persistence

        Returns:
            VectorStoreService: Configured vector store service instance
        """
        text_chunker = DummyTextChunker()
        vector_store_repo: VectorStoreRepository = ServerQdrantVectorStoreRepository(
            embedding, text_chunker,
            collection_name=dataset_folder.replace("/", "_")
        )
        return VectorStoreService(vector_store_repo)

        
    def _clean_if_incomplete(dataset_folder: str) -> tuple[dict[str, str], dict[str, str], bool]:
        """
        Clean incomplete vector store or load existing mappings.

        If the mapping pickle files exist, loads and returns them.
        Otherwise, cleans the dataset folder by removing all files.

        Args:
            dataset_folder (str): Path to dataset folder containing
                vector store

        Returns:
            tuple: (doc_to_memory mapping dict, memory_to_doc mapping dict, bool
                indicating if pickles were complete)
        """
        doc_to_memory_path = os.path.join(dataset_folder, "doc_to_memory.pkl")
        memory_to_doc_path = os.path.join(dataset_folder, "memory_to_doc.pkl")

        if os.path.exists(doc_to_memory_path) and os.path.exists(memory_to_doc_path):
            with open(doc_to_memory_path, "rb") as f:
                doc_to_memory = pickle.load(f)
            with open(memory_to_doc_path, "rb") as f:
                memory_to_doc = pickle.load(f)
            return doc_to_memory, memory_to_doc, True
        
        doc_to_memory_unfinished_path = os.path.join(dataset_folder, "doc_to_memory_unfinished.pkl")
        memory_to_doc_unfinished_path = os.path.join(dataset_folder, "memory_to_doc_unfinished.pkl")
        if os.path.exists(doc_to_memory_unfinished_path) or os.path.exists(memory_to_doc_unfinished_path):
            with open(doc_to_memory_unfinished_path, "rb") as f:
                try:
                    doc_to_memory = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading unfinished doc_to_memory.pkl: {e}")
                    doc_to_memory = {}
                logger.info(f"Loaded {len(doc_to_memory)} entries from unfinished doc_to_memory.pkl")
            with open(memory_to_doc_unfinished_path, "rb") as f:
                try:
                    memory_to_doc = pickle.load(f)
                    logger.info(f"Loaded {len(memory_to_doc)} entries from unfinished memory_to_doc.pkl")
                except Exception as e:
                    logger.error(f"Error loading unfinished memory_to_doc.pkl: {e}")
                    memory_to_doc = {}
            if len(doc_to_memory) == 0 and len(memory_to_doc) > 0:
                # recreate doc_to_memory from memory_to_doc
                doc_to_memory = {
                    dataset_doc_id: saved_memory_id
                    for saved_memory_id, dataset_doc_id in memory_to_doc.items()
                }
            elif len(memory_to_doc) == 0 and len(doc_to_memory) > 0:
                # recreate memory_to_doc from doc_to_memory
                memory_to_doc = {
                    saved_memory_id: dataset_doc_id
                    for dataset_doc_id, saved_memory_id in doc_to_memory.items()
                }
            if len(doc_to_memory) == len(memory_to_doc):            
                logger.info("Reconstructed mappings are consistent.")
                return doc_to_memory, memory_to_doc, False

        if os.path.exists(dataset_folder):
            # If pickles do not exist, clean the directory
            for file in os.listdir(dataset_folder):
                file_path = os.path.join(dataset_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        return {}, {}, False

    @classmethod
    async def create_filled_vector_store_service(
        cls,
        dataset: DataFrameDataset,
        dataset_folder: str,
        embedding: EmbeddingModel,
        batch_size: int = 1024,  # Process 1024 documents at a time
        db_batch_size: int = 512, 
    ) -> tuple[VectorStoreService, dict[str, str], dict[str, str]]:
        """
        Load dataset and create filled vector store service with mappings.

        Indexes all documents from the dataset into the vector store,
        creating bidirectional mappings between dataset document IDs
        and memory IDs. Persists mappings to pickle files for reuse.

        Args:
            dataset (DataFrameDataset): Dataset to load documents from
            dataset_folder (str): Path to folder for vector store
                persistence
            embedding (EmbeddingModel): Embedding model for vectorization

        Returns:
            tuple: (vector_store_service, doc_to_memory mapping,
                memory_to_doc mapping)
        """
        doc_to_memory, memory_to_doc, complete = cls._clean_if_incomplete(dataset_folder)
        vector_store_service = cls._create_vector_store_service(
            embedding, dataset_folder
        )

        if complete:
            logger.info("Pickle files are valid. Skipping ingestion loop.")
            return vector_store_service, doc_to_memory, memory_to_doc

        docs_df = dataset.docs[["id", "content"]]
        docs_list = list(docs_df.itertuples(index=False))

        # Print all already indexed documents
        print("Already indexed documents:" + str(len(doc_to_memory)))

        # Process in batches
        for i in tqdm(range(0, len(docs_list), batch_size), desc="Indexing batches"):
            batch = docs_list[i:i + batch_size]
            
            # Create MemoryRequest objects for this batch
            batch_memories = []
            batch_mappings = []
            
            for doc in batch:
                current_passage = MemoryRequest.create(text=[str(doc[1])])
                batch_memories.append(current_passage)
                
                dataset_doc_id = str(doc[0])
                
                if dataset_doc_id in doc_to_memory.keys():
                    logger.info(f"Document {dataset_doc_id} already indexed. Skipping.")
                    continue

                batch_mappings.append((dataset_doc_id, current_passage.id))
            
            try:
                # Batch index all memories at once
                await vector_store_service.index_memories_batch(batch_memories, qdrant_batch_size=db_batch_size)

                # Update mappings
                for dataset_doc_id, saved_memory_id in batch_mappings:
                    doc_to_memory[dataset_doc_id] = saved_memory_id
                    memory_to_doc[saved_memory_id] = dataset_doc_id
            except RuntimeError as e:
                logger.error(f"Error indexing batch: {e}")

                os.makedirs(dataset_folder, exist_ok=True)
                with open(os.path.join(dataset_folder, "doc_to_memory_unfinished.pkl"), "wb") as f:
                    pickle.dump(doc_to_memory, f)

                with open(os.path.join(dataset_folder, "memory_to_doc_unfinished.pkl"), "wb") as f:
                    pickle.dump(memory_to_doc, f)
            
            gc.collect()

        os.makedirs(dataset_folder, exist_ok=True)
        with open(os.path.join(dataset_folder, "doc_to_memory.pkl"), "wb") as f:
            pickle.dump(doc_to_memory, f)

        with open(os.path.join(dataset_folder, "memory_to_doc.pkl"), "wb") as f:
            pickle.dump(memory_to_doc, f)

        logger.info("Saved doc_to_memory and memory_to_doc to pickle files.")

        return vector_store_service, doc_to_memory, memory_to_doc

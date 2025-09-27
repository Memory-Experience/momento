import logging
import os
import pickle

from tqdm import tqdm

from dataset.dataset import DataFrameDataset
from api.vector_store.vector_store_service import VectorStoreService
from api.domain.memory_request import MemoryRequest
from api.vector_store.repositories.qdrant_vector_store_repository import (
    LocalFileQdrantVectorStoreRepository,
)
from api.vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)
from api.models.embedding.embedding_model_interface import EmbeddingModel

import gc

from baseline.dummy_text_chunker import DummyTextChunker


# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    def _create_vector_store_service(
        embedding: EmbeddingModel, dataset_folder
    ) -> VectorStoreService:
        text_chunker = DummyTextChunker()
        vector_store_repo: VectorStoreRepository = LocalFileQdrantVectorStoreRepository(
            embedding, text_chunker, dataset_folder
        )
        return VectorStoreService(vector_store_repo)

    def _clean_if_incomplete(dataset_folder) -> tuple[dict[str, str], dict[str, str]]:
        doc_to_memory_path = os.path.join(dataset_folder, "doc_to_memory.pkl")
        memory_to_doc_path = os.path.join(dataset_folder, "memory_to_doc.pkl")

        if os.path.exists(doc_to_memory_path) and os.path.exists(memory_to_doc_path):
            with open(doc_to_memory_path, "rb") as f:
                doc_to_memory = pickle.load(f)
            with open(memory_to_doc_path, "rb") as f:
                memory_to_doc = pickle.load(f)
            return doc_to_memory, memory_to_doc

        if os.path.exists(dataset_folder):
            # If pickles do not exist, clean the directory
            for file in os.listdir(dataset_folder):
                file_path = os.path.join(dataset_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        return {}, {}

    @classmethod
    async def create_filled_vector_store_service(
        cls,
        dataset: DataFrameDataset,
        dataset_folder: str,
        embedding: EmbeddingModel,
    ) -> tuple[VectorStoreService, dict[str, str], dict[str, str]]:
        doc_to_memory, memory_to_doc = cls._clean_if_incomplete(dataset_folder)
        vector_store_service = cls._create_vector_store_service(
            embedding, dataset_folder
        )

        if len(doc_to_memory) > 0 and len(doc_to_memory) == len(memory_to_doc):
            logger.info("Pickle files are valid. Skipping ingestion loop.")
            return vector_store_service, doc_to_memory, memory_to_doc

        docs_df = dataset.docs[["id", "content"]]  # Keep only necessary columns

        for doc in tqdm(docs_df.itertuples(index=False), total=len(docs_df)):
            current_passage = MemoryRequest.create(text=[str(doc[1])])

            await vector_store_service.index_memory(current_passage)

            dataset_doc_id = str(doc[0])
            saved_memory_id = current_passage.id

            doc_to_memory[dataset_doc_id] = saved_memory_id
            memory_to_doc[saved_memory_id] = dataset_doc_id

            gc.collect()

        with open(os.path.join(dataset_folder, "doc_to_memory.pkl"), "wb") as f:
            pickle.dump(doc_to_memory, f)

        with open(os.path.join(dataset_folder, "memory_to_doc.pkl"), "wb") as f:
            pickle.dump(memory_to_doc, f)

        logger.info("Saved doc_to_memory and memory_to_doc to pickle files.")

        return vector_store_service, doc_to_memory, memory_to_doc

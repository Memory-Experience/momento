import logging

from tqdm import tqdm

from baseline.lucene_vector_store_repository import LuceneVectorStoreRepository
from api.vector_store.vector_store_service import VectorStoreService
from api.domain.memory_request import MemoryRequest
from dataset.dataset import DataFrameDataset
from dataset_loader import DatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BM25DatasetLoader(DatasetLoader):
    """
    Dataset loader that uses BM25 (Lucene) for retrieval.

    This loader creates a Lucene-based vector store repository
    instead of embedding-based retrieval, providing a BM25 baseline
    for evaluation comparisons.
    """

    @classmethod
    async def create_filled_vector_store_service(
        cls, dataset: DataFrameDataset, dataset_folder: str
    ) -> tuple[VectorStoreService, dict[str, str], dict[str, str]]:
        """
        Create filled vector store using Lucene BM25 index.

        Parameters:
            dataset (DataFrameDataset): Dataset to load documents from
            dataset_folder (str): Path to folder for Lucene index
                persistence

        Returns:
            tuple: (vector_store_service, doc_to_memory mapping,
                memory_to_doc mapping)
        """
        doc_to_memory, memory_to_doc = cls._clean_if_incomplete(dataset_folder)

        vector_store_repo = LuceneVectorStoreRepository(index_dir=dataset_folder)
        vector_store_service = VectorStoreService(vector_store_repo)

        docs_df = dataset.docs[["id", "content"]]

        for doc in tqdm(docs_df.itertuples(index=False), total=len(docs_df)):
            current_passage = MemoryRequest.create(text=[str(doc[1])])
            await vector_store_service.index_memory(current_passage)

            dataset_doc_id = str(doc[0])
            saved_memory_id = current_passage.id
            doc_to_memory[dataset_doc_id] = saved_memory_id
            memory_to_doc[saved_memory_id] = dataset_doc_id

        return vector_store_service, doc_to_memory, memory_to_doc

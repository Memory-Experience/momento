import logging
from uuid import UUID, uuid4

from pyproj import err

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
)

from ...models.embedding.embedding_model_interface import EmbeddingModel
from ...models.text_chunker_interface import TextChunker

from ...domain.memory_context import MemoryContext
from ...domain.memory_request import MemoryRequest, MemoryType

from .vector_store_repository_interface import (
    FilterArg,
    FilterCondition,
    FilterGroup,
    FilterOperator,
    VectorStoreRepository,
)


class QdrantVectorStoreRepository(VectorStoreRepository):
    """
    Repository implementation that stores memories and their
    vector embeddings using Qdrant in-memory mode.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedding_model: EmbeddingModel,
        text_chunker: TextChunker,
        collection_name="memories",
    ):
        """
        Initialize the Qdrant in-memory vector store repository.

        Args:
            collection_name: Name of the collection to store vectors
            vector_size: Dimension of vectors
                (should match your embedding model's output)
        """
        super().__init__(embedding_model=embedding_model, text_chunker=text_chunker)

        self.client = client
        self.collection_name = collection_name
        self.vector_size = embedding_model.get_vector_size()

        # Create collection for memories
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )

        # Store memory objects for quick access (not in vector store)
        self._memories: dict[UUID, MemoryRequest] = {}

        logging.info(
            "Initialized InMemoryVectorStoreRepository "
            f"with Qdrant (collection: {collection_name})"
        )

    async def index_memory(self, memory: MemoryRequest) -> None:
        """
        Index a memory using its embedding.

        Args:
            memory: The memory to index
        """
        memory_id = memory.id

        # Store the memory object for quick access
        self._memories[memory_id] = memory

        # Join the list of text segments into a single string
        full_text = " ".join(memory.text)

        # Generate chunks using the text chunker
        chunks = self.text_chunker.chunk_text(full_text)

        # Create points for each chunk and the full text
        points = []

        # First, add the full text embedding
        full_text_vector = await self.embedding_model.embed_text(full_text)

        # Prepare payload for vector DB
        metadata = {
            "text": memory.text,
            "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
            "memory_type": memory.memory_type.value,
            "is_chunk": False,  # Flag to indicate this is a full text embedding
            "parent_id": None,  # No parent for the full text
        }

        # Add full text point
        points.append(
            PointStruct(
                id=str(memory_id),
                vector=full_text_vector,
                payload={
                    **metadata,
                    "text_content": full_text,  # Store the actual content for retrieval
                },
            )
        )

        if len(chunks) > 1:
            # Now add points for each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid4())
                chunk_vector = await self.embedding_model.embed_text(chunk)

                chunk_metadata = {
                    **metadata,  # Copy the base metadata
                    "is_chunk": True,  # Flag to indicate this is a chunk
                    "parent_id": memory_id,  # Link back to the parent memory
                    "chunk_index": i,  # Store the index of this chunk
                    "text_content": chunk,  # Store the actual chunk text
                }

                points.append(
                    PointStruct(
                        id=chunk_id, vector=chunk_vector, payload=chunk_metadata
                    )
                )

        # Store all points in Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

        logging.info(f"Indexed memory {memory_id} with {len(chunks)} chunks")

    async def index_memories_batch(
        self, memories: list[MemoryRequest], qdrant_batch_size: int = 512
    ) -> None:
        """
        Index multiple memories in batch for better performance.

        Args:
            memories: List of memories to index
        """
        if not memories:
            return

        # Store all memories in cache
        for memory in memories:
            self._memories[memory.id] = memory

        # Collect all texts that need embedding
        texts_to_embed = []
        text_metadata = []  # Track what each text is for

        for memory in memories:
            full_text = " ".join(memory.text)
            chunks = self.text_chunker.chunk_text(full_text)

            # Add full text
            texts_to_embed.append(full_text)
            text_metadata.append({
                "type": "full",
                "memory": memory,
                "full_text": full_text,
                "chunks": chunks,
            })

            # Add chunks if there are multiple
            if len(chunks) > 1:
                for i, chunk in enumerate(chunks):
                    texts_to_embed.append(chunk)
                    text_metadata.append({
                        "type": "chunk",
                        "memory": memory,
                        "chunk_text": chunk,
                        "chunk_index": i,
                        "full_text": full_text,
                    })

        # Batch embed all texts at once
        if hasattr(self.embedding_model, "embed_texts_batch"):
            embeddings = await self.embedding_model.embed_texts_batch(texts_to_embed)
        else:
            # Fallback to individual embeddings if batch method not available
            embeddings = [
                await self.embedding_model.embed_text(text) for text in texts_to_embed
            ]

        # Build points for Qdrant
        points = []
        for i, (metadata, embedding) in enumerate(
            zip(text_metadata, embeddings, strict=True)
        ):
            memory = metadata["memory"]

            base_payload = {
                "text": memory.text,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                "memory_type": memory.memory_type.value,
            }

            if metadata["type"] == "full":
                points.append(
                    PointStruct(
                        id=str(memory.id),
                        vector=embedding,
                        payload={
                            **base_payload,
                            "is_chunk": False,
                            "parent_id": None,
                            "text_content": metadata["full_text"],
                        },
                    )
                )
            else:  # chunk
                chunk_id = str(uuid4())
                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            **base_payload,
                            "is_chunk": True,
                            "parent_id": memory.id,
                            "chunk_index": metadata["chunk_index"],
                            "text_content": metadata["chunk_text"],
                        },
                    )
                )

            if i % qdrant_batch_size == 0 and i > 0:
                # Batch upsert to Qdrant
                # make a small try catch loop to handle potential upsert errors
                retries = 5
                while points:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name, points=points
                        )
                        points = []  # Clear points after upsert
                    except Exception as e:
                        logging.warning(f"Error upserting to Qdrant: {e}")

                        if retries == 0:
                            logging.error(
                                "Max retries reached. Some points may not be indexed."
                            )
                            raise RuntimeError(
                                "Failed to upsert points to"
                                " Qdrant after multiple attempts."
                            ) from err

        # Batch upsert to Qdrant
        if points:
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []  # Clear points after upsert
            except Exception as e:
                logging.warning(f"Error upserting to Qdrant: {e}")

        logging.info(
            f"Batch indexed {len(memories)} memories with {len(points)} total points"
        )

    async def get_memory(self, memory_id: UUID) -> MemoryRequest | None:
        """
        Get a memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            The Memory if found, None otherwise
        """
        # First check our local cache
        if memory_id in self._memories:
            return self._memories[memory_id]

        # If not in cache, try to get from Qdrant
        points = self.client.retrieve(
            collection_name=self.collection_name, ids=[memory_id]
        )

        if not points:
            return None

        # Reconstruct memory from Qdrant point
        point = points[0]

        # Check if this is a chunk and we need to fetch the parent
        is_chunk = point.payload.get("is_chunk", False)
        if is_chunk:
            parent_id = point.payload.get("parent_id")
            if parent_id:
                return await self.get_memory(parent_id)

        # Create memory instance with required fields
        memory = MemoryRequest.create(
            id=UUID(memory_id),
            text=point.payload.get("text", []),
            memory_type=MemoryType(point.payload.get("memory_type", 0)),
            timestamp=point.payload.get("timestamp"),
        )

        # Update cache
        self._memories[memory_id] = memory
        return memory

    async def search_similar(
        self,
        query: MemoryRequest,
        limit: int = 5,
        filters: FilterArg = None,
        search_chunks: bool = True,
    ) -> MemoryContext:
        """
        Search for memories with content similar to the query embedding.

        Uses vector similarity search with optional filtering.

        Args:
            query: The query memory to search with
            limit: Maximum number of results to return
            filters: Optional filters to apply
            search_chunks: Whether to search in chunks (True) or only full texts (False)

        Returns:
            MemoryContext containing the search results
        """
        query_text = " ".join(query.text)
        query_vector = await self.embedding_model.embed_text(query_text)

        query_filter = self._create_search_filter(filters, search_chunks)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        context = MemoryContext.create(query)

        for result in search_results.points:
            memory, matched_text = self._get_or_create_memory(result)

            logging.info(f"Found memory {memory.text} with score: {result.score:.4f}")

            context.add_memory(
                memory=memory,
                score=result.score,
                matched_text=matched_text,
            )

        return context

    def _create_search_filter(
        self, filters: FilterArg, search_chunks: bool
    ) -> Filter | None:
        """
        Create a search filter combining user
        filters and chunk filtering if needed.
        """
        # Convert user filters to Qdrant format
        qdrant_filter = self._convert_filter_to_qdrant(filters) if filters else None

        # If we're not searching chunks, add a filter to exclude them
        if not search_chunks:
            chunk_filter = Filter(
                must=[FieldCondition(key="is_chunk", match=MatchValue(value=False))]
            )

            # Combine with any existing filters
            if qdrant_filter:
                return Filter(must=[qdrant_filter, chunk_filter])
            else:
                return chunk_filter

        return qdrant_filter

    def _get_or_create_memory(self, result) -> tuple[MemoryRequest, str]:
        """Get a memory from cache or create it from search result."""
        result_id = result.id
        is_chunk = result.payload.get("is_chunk", False)
        matched_text = result.payload.get("text_content", "")

        # Determine memory ID (parent for chunks, direct for full texts)
        if is_chunk:
            memory_id = result.payload.get("parent_id")
            # Try cache first
            if memory_id in self._memories:
                return self._memories[memory_id], matched_text

            # Fetch parent memory
            parent_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
            )

            if parent_points:
                parent = parent_points[0]
                memory = MemoryRequest.create(
                    id=UUID(memory_id),
                    text=parent.payload.get("text", []),
                    memory_type=MemoryType(parent.payload.get("memory_type", 0)),
                    timestamp=parent.payload.get("timestamp"),
                )
                self._memories[memory_id] = memory
                return memory, matched_text
            else:
                # Fallback if parent not found
                memory = MemoryRequest.create(
                    id=UUID(memory_id),
                    text=[matched_text],
                    memory_type=MemoryType(result.payload.get("memory_type", 0)),
                )
                return memory, matched_text
        else:
            # Direct memory match
            memory_id = result_id
            if memory_id in self._memories:
                return self._memories[memory_id], matched_text

            memory = MemoryRequest.create(
                id=UUID(memory_id),
                text=result.payload.get("text", []),
                memory_type=MemoryType(result.payload.get("memory_type", 0)),
                timestamp=result.payload.get("timestamp"),
            )
            self._memories[memory_id] = memory
            return memory, matched_text

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete all information for a memory, including its chunks.

        Args:
            memory_id: The ID of the memory to delete
        """
        # First, find and delete all chunks associated with this memory
        chunk_filter = Filter(
            must=[FieldCondition(key="parent_id", match=MatchValue(value=memory_id))]
        )

        # Find all chunks
        chunk_results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=chunk_filter,
            limit=1000,
            with_payload=False,
            with_vectors=False,
        )

        # Get chunk IDs
        chunk_ids = [point.id for point in chunk_results]

        # Delete all chunks if any were found
        if chunk_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=chunk_ids,  # Use the IDs directly
            )
            logging.info(f"Deleted {len(chunk_ids)} chunks for memory {memory_id}")

        # Delete the main memory point
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[memory_id],  # Use a list with the ID
        )

        # Also remove from local cache if present
        if memory_id in self._memories:
            del self._memories[memory_id]

        logging.info(f"Deleted memory {memory_id} from vector store")

    async def list_memories(
        self, limit: int = 100, offset: UUID | None = None, filters: FilterArg = None
    ) -> tuple[list[MemoryRequest], UUID]:
        """
        List memories in the vector store.

        Args:
            limit: Maximum number of memories to return
            offset: If provided, skip memories with ids less than this value
            filters: Optional filters to apply using the abstract filter system

        Returns:
            List of Memory objects
        """
        # Create base filter to exclude chunks
        chunk_filter = Filter(
            must=[FieldCondition(key="is_chunk", match=MatchValue(value=False))]
        )

        # Convert user filters to Qdrant format if provided
        user_filter = self._convert_filter_to_qdrant(filters) if filters else None

        # Combine filters
        if user_filter:
            combined_filter = Filter(must=[chunk_filter, user_filter])
        else:
            combined_filter = chunk_filter

        # Use Qdrant's scroll method for pagination
        result, next_page_offset = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=combined_filter,
            limit=limit,
            offset=offset,
            with_payload=True,  # We need the payload to reconstruct the Memory objects
            with_vectors=False,  # We don't need the vectors for listing
        )

        memories = []
        for point in result:
            memory_id = UUID(point.id)

            # Use cached memory if available, otherwise reconstruct
            if memory_id in self._memories:
                memory = self._memories[memory_id]
                memories.append(memory)
            else:
                # Create memory from point payload
                memory = MemoryRequest.create(
                    id=memory_id,
                    text=point.payload.get("text", []),
                    memory_type=MemoryType(point.payload.get("memory_type", 0)),
                    timestamp=point.payload.get("timestamp"),
                )
                # Update cache
                self._memories[memory_id] = memory
                memories.append(memory)

        logging.info(f"Listed {len(memories)} memories")
        return memories, next_page_offset

    def _convert_filter_to_qdrant(self, filter_arg: FilterArg) -> Filter | None:
        """
        Convert the abstract filter condition to Qdrant's filter format.

        Args:
            filter_arg: Abstract filter condition

        Returns:
            Equivalent Qdrant filter
        """
        if not filter_arg:
            return None

        if isinstance(filter_arg, FilterCondition):
            # Handle single filter condition
            return self._convert_single_condition_to_qdrant(filter_arg)
        elif isinstance(filter_arg, FilterGroup):
            # Handle filter group (AND/OR)
            conditions = []

            for condition in filter_arg.conditions:
                converted = self._convert_filter_to_qdrant(condition)
                if converted:
                    conditions.append(converted)

            if not conditions:
                return None

            if filter_arg.operator.upper() == "AND":
                return Filter(must=conditions)
            else:  # OR
                return Filter(should=conditions)

        return None

    def _convert_single_condition_to_qdrant(self, condition: FilterCondition) -> Filter:
        """
        Convert a single filter condition to Qdrant's filter format.

        Args:
            condition: Filter condition to convert

        Returns:
            Qdrant filter
        """
        # Get field path - handle metadata fields specially
        field = condition.field
        if not field.startswith("metadata."):
            field = f"metadata.{field}"

        # Convert based on the operator
        if condition.operator == FilterOperator.EQUALS:
            field_condition = FieldCondition(
                key=field, match=MatchValue(value=condition.value)
            )
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.NOT_EQUALS:
            field_condition = FieldCondition(
                key=field, match=MatchValue(value=condition.value)
            )
            return Filter(must_not=[field_condition])

        elif condition.operator == FilterOperator.GREATER_THAN:
            field_condition = FieldCondition(key=field, range=Range(gt=condition.value))
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
            field_condition = FieldCondition(
                key=field, range=Range(gte=condition.value)
            )
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.LESS_THAN:
            field_condition = FieldCondition(key=field, range=Range(lt=condition.value))
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.LESS_THAN_OR_EQUAL:
            field_condition = FieldCondition(
                key=field, range=Range(lte=condition.value)
            )
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.EXISTS:
            field_condition = FieldCondition(
                key=field, match=models.IsNullCondition(is_null=False)
            )
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.NOT_EXISTS:
            field_condition = FieldCondition(
                key=field, match=models.IsNullCondition(is_null=True)
            )
            return Filter(must=[field_condition])

        elif condition.operator == FilterOperator.CONTAINS:
            # For contains, we could use a more complex match in a real implementation
            # This is a simplified version that just checks for equality
            field_condition = FieldCondition(
                key=field, match=MatchValue(value=condition.value)
            )
            return Filter(must=[field_condition])

        # Default case - no valid conversion
        logging.warning(f"Unsupported filter operator: {condition.operator}")
        return None


class InMemoryQdrantVectorStoreRepository(QdrantVectorStoreRepository):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        text_chunker: TextChunker,
        collection_name="memories",
    ):
        """
        Initialize the Qdrant in-memory vector store repository.

        Args:
            embedding_model: The embedding model to use for vectorizing text
            text_chunker: The text chunker to use for splitting text into chunks
            collection_name: Name of the collection to store vectors
        """

        # Initialize in-memory Qdrant client
        client = QdrantClient(":memory:")

        super().__init__(
            client=client,
            embedding_model=embedding_model,
            text_chunker=text_chunker,
            collection_name=collection_name,
        )


class LocalFileQdrantVectorStoreRepository(QdrantVectorStoreRepository):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        text_chunker: TextChunker,
        database_path: str,
        collection_name="memories",
    ):
        """
        Initialize the Qdrant file-based vector store repository.

        Args:
            embedding_model: The embedding model to use for vectorizing text
            text_chunker: The text chunker to use for splitting text into chunks
            database_path: Path to the local database file
            collection_name: Name of the collection to store vectors
        """

        # Initialize in-memory Qdrant client
        client = QdrantClient(path=database_path)

        super().__init__(
            client=client,
            embedding_model=embedding_model,
            text_chunker=text_chunker,
            collection_name=collection_name,
        )


class ServerQdrantVectorStoreRepository(QdrantVectorStoreRepository):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        text_chunker: TextChunker,
        url: str = "http://localhost:6333",
        collection_name="memories",
    ):
        """
        Initialize the Qdrant in-memory vector store repository.

        Args:
            collection_name: Name of the collection to store vectors
            vector_size: Dimension of vectors
                (should match your embedding model's output)
        """

        # Initialize in-memory Qdrant client
        client = QdrantClient(url=url)

        super().__init__(
            client=client,
            embedding_model=embedding_model,
            text_chunker=text_chunker,
            collection_name=collection_name,
        )

import logging
from collections.abc import AsyncIterator

from protos.generated.py import stt_pb2

from .domain.memory_request import MemoryRequest, MemoryType
from .dependency_container import Container


class QuestionAnswerService:
    """
    Service for answering questions using RAG (Retrieval-Augmented Generation).

    This service handles question answering by:
    1. Receiving streaming question input (audio or text)
    2. Searching for relevant memories in the vector store
    3. Filtering results by relevance threshold
    4. Generating answers using an LLM with the retrieved context
    5. Streaming the answer back to the client
    """

    def __init__(self, dependencies: Container):
        """
        Initialize the question answer service.

        Args:
            dependencies: Container with all required service dependencies including
                         vector_store, persistence, rag, and threshold_filter services
        """
        self.vector_store_service = dependencies.vector_store
        self.persistence_service = dependencies.persistence
        self.rag_service = dependencies.rag
        self.retrieval_limit = dependencies.retrieval_limit
        self.threshold_filter_service = dependencies.threshold_filter

    async def AnswerQuestion(
        self, request_iterator, context
    ) -> AsyncIterator[stt_pb2.MemoryChunk]:
        """
        Answer questions from a stream of MemoryChunk messages.

        This gRPC method processes streaming question input, retrieves relevant
        memories, and generates a streaming answer using RAG.

        Parameters:
            request_iterator (AsyncIterator[MemoryChunk]): Async iterator
                yielding MemoryChunk protobuf messages
            context (grpc.aio.ServicerContext): gRPC context for the
                request

        Yields:
            protos.generated.py.stt_pb2.MemoryChunk: Stream of memory
                context and answer chunks
        """
        session_id = None
        memory_id = None
        audio_data = bytearray()
        transcription = []

        async for chunk in request_iterator:
            logging.debug(f"<<<Received question chunk: {chunk}")

            # Extract metadata
            if chunk.metadata:
                session_id = chunk.metadata.session_id or session_id
                memory_id = chunk.metadata.memory_id or memory_id

            # Accumulate audio data and transcription text
            if chunk.audio_data:
                audio_data.extend(chunk.audio_data)
            if chunk.text_data:
                transcription.append(chunk.text_data)

            # If this is the final chunk, process and answer the question
            if chunk.metadata and chunk.metadata.is_final:
                async for response in self._process_question(
                    audio_data, transcription, session_id, memory_id
                ):
                    yield response

                # Reset for next question within the same connection
                audio_data = bytearray()
                transcription = []
                session_id = None
                memory_id = None

    async def _process_question(self, audio_data, transcription, session_id, memory_id):
        """
        Process a question, fetch context, and stream answer.

        Args:
            audio_data: Raw audio bytes (if question was spoken)
            transcription: List of transcribed text segments forming the question
            session_id: Session identifier for the client connection
            memory_id: Question identifier from the client

        Yields:
            MemoryChunk: Stream of relevant memories followed by generated answer chunks
        """
        full_transcription = " ".join(transcription)
        logging.info(f"Processing question: {full_transcription}")

        # Create question memory object
        question_memory = MemoryRequest.create(
            audio_data=bytes(audio_data) if audio_data else None,
            text=transcription,
            memory_type=MemoryType.QUESTION,
        )

        # Get memory context
        memory_context = await self.vector_store_service.search(
            question_memory, limit=self.retrieval_limit
        )

        # Apply threshold filtering for better precision
        filtered_context = self.threshold_filter_service.filter_context(memory_context)

        # Generate answer
        response_generator = self.rag_service.answer_question(
            query=question_memory,
            memory_context=filtered_context,
            chunk_size_tokens=8,
        )

        # Stream memory context first if available
        if filtered_context and filtered_context.memories:
            memory_count = len(filtered_context.memories)
            logging.info(f"Sending {memory_count} filtered memories from context")
            for memory in filtered_context.memories.values():
                # Create the memory chunk using the factory method
                memory_chunk = memory.to_chunk(
                    session_id=session_id, chunk_type=stt_pb2.ChunkType.MEMORY
                )

                # Then add the score from filtered_context
                if memory.id in filtered_context.scores:
                    memory_chunk.metadata.score = float(
                        filtered_context.scores[memory.id]
                    )

                yield memory_chunk

        # Stream answer chunks
        try:
            last_chunk = None
            async for response_chunk in response_generator:
                if response_chunk.response and response_chunk.response.text:
                    chunks_count = len(response_chunk.response.text)
                    for i, text_segment in enumerate(response_chunk.response.text):
                        if text_segment.strip():
                            is_final = (
                                i == chunks_count - 1
                                and response_chunk.metadata.get("is_final", False)
                            )

                            answer_chunk = stt_pb2.MemoryChunk(
                                text_data=text_segment,
                                metadata=stt_pb2.ChunkMetadata(
                                    session_id=session_id,
                                    memory_id=str(response_chunk.response.id),
                                    type=stt_pb2.ChunkType.ANSWER,
                                    is_final=is_final,
                                ),
                            )
                            last_chunk = answer_chunk
                            yield answer_chunk

            # If the generator finishes and the last chunk sent was not final,
            # -> send a final marker.
            if last_chunk and not last_chunk.metadata.is_final:
                final_marker = stt_pb2.MemoryChunk(
                    metadata=stt_pb2.ChunkMetadata(
                        session_id=session_id,
                        memory_id=last_chunk.metadata.memory_id,
                        type=stt_pb2.ChunkType.ANSWER,
                        is_final=True,
                    )
                )
                yield final_marker
                logging.info("Sent final answer marker because stream ended.")

            logging.info("Answer streaming completed")
        except Exception as e:
            logging.error(f"Error streaming answer: {e}")
            raise e

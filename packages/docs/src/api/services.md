# Core Services

The API package organizes business logic into focused service components, each with a single, well-defined responsibility. The service layer exemplifies clean architecture principles: services depend on abstractions (interfaces) rather than concrete implementations, enabling testability, flexibility, and clear separation of concerns.

## VectorStoreService

**Location**: `vector_store/vector_store_service.py`

Manages semantic search and memory indexing operations, providing a high-level interface over the vector database.

### Key Design Decisions

**Repository delegation**: The service delegates all chunking, embedding, and storage concerns to the `VectorStoreRepository`. This keeps the service focused on orchestration: it doesn't know whether embeddings come from Qwen3, SBert, or OpenAI, nor whether storage uses QDrant, Pinecone, or Weaviate.

**Domain model interface**: Methods accept and return domain objects (`MemoryRequest`, `MemoryContext`) rather than primitive types. This provides type safety and encapsulates business rules. For example, `MemoryRequest.create()` ensures all memories have valid IDs and timestamps.

**Async operations**: All methods are async, enabling non-blocking I/O. Vector operations can be slow (especially embedding generation), so async design prevents blocking the event loop and allows the server to handle other requests concurrently.

### Repository Interface

The service depends on `VectorStoreRepository`, which defines methods like `index_memory()`, `search_similar()`, and `delete_memory()`. Concrete implementations handle:

- Text chunking (sentence-level, character-level, or semantic)
- Embedding generation (via injected embedding models)
- Vector storage (QDrant, Pinecone, Weaviate, etc.)
- Metadata management (linking chunks back to original memories)

This abstraction exemplifies the **Repository Pattern**: the service treats the vector store as a collection of domain objects, not as a database with queries.

## PersistenceService

**Location**: `persistence/persistence_service.py`

Handles file-based storage of memory metadata and audio data.

### Key Design Decisions

**File-based storage**: The system uses JSON files rather than a database for memory persistence. This choice prioritizes simplicity and transparency: users can easily inspect, back up, or version control their memories. The trade-off is scalability (suitable for hundreds of memories, not millions) and lack of complex querying.

**Repository abstraction**: Like `VectorStoreService`, persistence is abstracted behind a repository interface. The `FileRepository` implementation handles serialization, but the service layer doesn't care, it could just as easily be `S3Repository` or `PostgresRepository`.

**Audio storage**: Audio is stored as base64-encoded data within JSON files. This keeps everything in one place and avoids managing separate file hierarchies. For larger audio files or higher throughput scenarios, separate blob storage (S3, local filesystem) would be more appropriate.

### Separation of Concerns

The service orchestrates persistence operations but delegates serialization details to the repository. This keeps format concerns (JSON structure, base64 encoding, file paths) isolated from business logic (save, load, delete operations).

## LLMRAGService

**Location**: `rag/llm_rag_service.py`

Implements retrieval-augmented generation (RAG) for answering questions using retrieved memory context.

### Key Design Decisions

**Thin service layer**: The service is deliberately thin: it primarily formats prompts and passes through the LLM's streaming output. This reflects a design choice to keep prompt engineering concerns (how to structure context, system messages, etc.) close to the LLM interaction rather than spread across multiple layers.

**Streaming pass-through**: Rather than accumulating tokens and sending complete responses, the service yields tokens as the LLM generates them. This zero-buffering approach minimizes latency and provides immediate user feedback. The trade-off is that error handling becomes more complex (errors can occur mid-stream).

**Model-agnostic interface**: The service depends on `LLMModel` interface, not a specific implementation. This enables A/B testing different models or comparing performance without changing service code: just inject a different model via the container.

### Prompt Engineering

The service constructs prompts that include:

- System message with role definition
- Retrieved memories formatted as numbered context
- Explicit instruction to ground answers in provided memories
- The user's question

This structure aims to reduce hallucination by making the grounding requirement explicit, though it's not foolproof (hence the additional `ThresholdFilterService` safeguard).

## ThresholdFilterService

**Location**: `rag/threshold_filter_service.py`

Filters search results by relevance score to improve answer precision and reduce hallucination risk.

### Key Design Decisions

**Quality over quantity**: Vector search returns the top K nearest neighbors, but "nearest" doesn't always mean "relevant." This service adds a quality gate, only memories exceeding a similarity threshold are passed to the LLM. This prevents the LLM from seeing marginally-related memories that could lead to confused or hallucinated answers.

**Stateless filtering**: The service is stateless and synchronous (unlike other services). It simply filters a `MemoryContext` and returns a new one. This simplicity reflects its focused responsibility, it's a pure function with no side effects or external dependencies.

**Configurable threshold**: The threshold represents a design trade-off between precision and recall. Higher thresholds reduce false positives (irrelevant memories) but increase false negatives (missing relevant memories). Currently set to 0 (disabled), meaning all top-K results are passed through without filtering. This decision reflects that similarity scores vary significantly depending on the question, making a static threshold unreliable. Instead, retrieval quality is controlled through the top-K limit alone.

### Rationale

**Why have the service if threshold is 0?** While currently disabled, the service architecture remains in place for experimentation and future improvements. Dynamic threshold selection or query-specific filtering could be implemented without changing the service composition.

**Why separate service?** Filtering could be done in `QuestionAnswerService`, but extracting it:

- Makes the filtering policy explicit and configurable
- Enables reuse (filtering might be useful elsewhere)
- Follows single responsibility principle
- Simplifies testing (easy to verify filtering logic in isolation)
- Allows toggling filtering on/off without code changes

## Service Composition

Services are composed to implement higher-level operations. These flows demonstrate how single-responsibility services combine to achieve complex behaviors.

### Question Answering Flow

```
Question → VectorStoreService (search) → ThresholdFilterService (filter)
→ LLMRAGService (generate) → Streaming response
```

This pipeline demonstrates the **Single Responsibility Principle** in action: each service handles one concern, and they're composed to achieve the desired behavior.

### Memory Storage Flow

```
Memory → PersistenceService (save to file)
Memory → VectorStoreService (index in vector DB)
```

Note the separation: persistence and indexing are independent operations. Memories can be persisted without indexing (useful for archival) or indexed without file storage (useful for ephemeral data).

## Service Layer Benefits

The service architecture provides several advantages:

**Testability**: Services depend on interfaces, making it trivial to inject mocks or test doubles. Testing service logic doesn't require real databases, models, or file systems.

**Composability**: Services are building blocks. Different compositions enable different features. The evaluation package reuses services in different configurations to compare retrieval strategies.

**Clear boundaries**: Each service has a well-defined interface and responsibility. This makes the codebase easier to understand and modify. Changes to vector store implementation don't affect persistence logic.

**Configuration flexibility**: Services are assembled via dependency injection (see below).

## Dependency Injection

Services are wired together using a container-based dependency injection pattern implemented in `dependency_container.py`. The `Container` dataclass holds all configured services and dependencies:

```python
@dataclass
class Container:
    vector_store: VectorStoreService
    persistence: PersistenceService
    rag: LLMRAGService
    threshold_filter: ThresholdFilterService
    transcriber: TranscriberInterface
```

### Construction Pattern

The container is created once at startup via `Container.create()`, which:

1. Creates leaf dependencies (models, chunkers, repositories)
2. Composes services (injecting their dependencies)
3. Returns the assembled container

Services receive dependencies through constructor injection, making dependencies explicit and enabling easy testing with mocks. The container is passed to WebSocket handlers, which extract the services they need.

### Key Benefits

**Explicit dependencies**: Reading a service's constructor reveals exactly what it needs. No hidden global state or framework magic.

**Testability**: Dependencies are injected, so tests can substitute mocks without touching real databases or models.

**Flexibility**: Different container configurations enable different system behaviors (development vs. production, or multiple evaluation configurations) without changing service code.

## Architecture Patterns

The service layer implementation follows several established architectural patterns that work together to create a maintainable and testable codebase.

### Repository Pattern

Data access is abstracted behind repository interfaces (`VectorStoreRepository`, `FileRepository`). Services interact with repositories through interfaces, remaining agnostic to underlying storage mechanisms. This enables:

- **Storage independence**: Swap QDrant for Pinecone, or files for PostgreSQL, without changing service code
- **Testing**: Mock repositories for unit testing without touching real databases
- **Multiple backends**: Run different storage implementations simultaneously (useful for evaluation)

### Domain-Driven Design

Services operate on rich domain models (`MemoryRequest`, `MemoryContext`) rather than primitive types or DTOs. These domain objects encapsulate business logic and provide factory methods, keeping services focused on orchestration rather than data manipulation.

### Single Responsibility Principle

Each service has one clear purpose:

- **VectorStoreService**: Semantic search and memory indexing
- **PersistenceService**: File-based memory storage
- **LLMRAGService**: Retrieval-augmented answer generation
- **ThresholdFilterService**: Relevance-based memory filtering

This granularity enables composability: services are building blocks that can be combined in different ways for different use cases.

### Dependency Injection

Services are wired together using a container-based dependency injection pattern implemented in `dependency_container.py`. The `Container` dataclass holds all configured services and dependencies, which are created once at startup and injected into services through constructor injection. This makes dependencies explicit, improves testability, and provides flexibility to configure different system behaviors without changing service code.

```

```

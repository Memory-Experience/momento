# API Endpoints

The API provides three WebSocket endpoints for real-time bidirectional communication and a REST endpoint for health checks.

## WebSocket Endpoints

The three WebSocket endpoints handle real-time communication using Protocol Buffers for efficient binary serialization.

### Overview

All endpoints follow a consistent pattern:

1. **Client connects** to a WebSocket endpoint
2. **Client sends chunks** (audio/text data with metadata)
3. **Server processes** and streams responses back
4. **Client/Server sends final marker** to complete the operation
5. **Client closes connection** to ensure scalability

## Protocol Design

All messages use Protocol Buffers (`MemoryChunk` message) for binary serialization. This design choice provides:

**Efficiency**: Binary format is more compact than JSON, reducing bandwidth for audio streaming
**Type safety**: Schema-defined messages prevent structural errors
**Versioning**: Protobuf's forward/backward compatibility enables API evolution
**Language agnostic**: Same schema works for Python backend and TypeScript frontend

The message schema uses a `oneof` field for data (either audio bytes or text string) and includes metadata for session tracking, memory identification, and operation control.

## Endpoint: `/ws/transcribe`

**Purpose**: Real-time speech-to-text transcription with support for both audio streaming and direct text input.

### Design Decisions

**Bidirectional streaming**: Rather than accumulating all audio before transcription, the endpoint streams results incrementally. This provides immediate feedback to users and enables real-time corrections.

**Audio buffering strategy**: Audio is accumulated in ~4-second windows before transcription, balancing latency against accuracy. Smaller windows would feel more responsive but provide less context for the model, degrading accuracy.

**Overlap handling**: A small overlap (0.1s) is maintained between chunks to prevent word breaks at arbitrary boundaries. This ensures continuity in transcription quality.

**Hybrid input support**: The endpoint accepts both audio and text, enabling typed input without requiring a separate code path. This flexibility supports accessibility and use cases where voice input isn't available.

**Explicit completion**: Clients send an `is_final` marker to signal completion rather than relying on connection closure. This enables multiple transcription operations over a single WebSocket connection.

## Endpoint: `/ws/memory`

**Purpose**: Store memories with automatic embedding generation and vector indexing.

### Design Decisions

**Dual storage**: Memories are persisted in both file storage (JSON) and vector storage (QDrant). This separation reflects different concerns: files provide durable, inspectable storage, while vectors enable semantic search.

**Atomic operations**: The final marker triggers both persistence and indexing as a logical unit. While not transactional in the database sense, this ensures memories are either fully saved or not saved at all (no partial states).

**Chunking at storage time**: Text is split into sentences during indexing, not during input. This keeps the client simple (no need to understand chunking strategy) and enables server-side experimentation with different chunking approaches.

**Background indexing**: While file save is fast, embedding generation and vector indexing can take seconds for long memories. The async processing model prevents blocking: the server can handle other requests while indexing completes.

**Memory ID generation**: The server assigns UUIDs rather than accepting client-provided IDs. This prevents ID conflicts and ensures uniqueness across the system.

## Endpoint: `/ws/ask`

**Purpose**: Answer questions using retrieval-augmented generation (RAG) over stored memories.

### Design Decisions

**Context-first response**: The endpoint streams relevant memories to the client _before_ generating the answer. This provides transparency: users see what context informed the answer, enabling them to judge reliability.

**Progressive disclosure**: Rather than waiting for the complete answer before displaying anything, the system streams tokens incrementally. This improves perceived performance and allows users to start reading while generation continues.

**Threshold filtering**: Not all top-K results from vector search are necessarily relevant. The threshold filter (default: 0.7) acts as a quality gate, preventing marginal matches from confusing the LLM or misleading users.

**Explicit grounding**: The system prompt explicitly instructs the LLM to answer only from provided memories. This is enforced at the prompt level, though threshold filtering provides an additional safeguard by ensuring only high-quality context reaches the LLM.

**Configurable granularity**: Token chunk size (default: 8) balances responsiveness against smoothness. This is exposed as a parameter because optimal values depend on UI design and user preference: some UIs look better with word-level streaming, others with phrase-level.

### RAG Pipeline Architecture

The endpoint implements a three-stage pipeline:

1. **Retrieval**: Vector search finds semantically similar memories
2. **Filtering**: Threshold-based quality gate
3. **Generation**: LLM synthesizes answer from filtered context

This separation of concerns makes each stage independently tunable and testable.

## Endpoint: `/health`

**Purpose**: Connectivity check for the frontend to verify backend reachability and display user-friendly error messages when the service is unavailable.

### Design Decisions

**Simple GET endpoint**: Returns a static JSON response `{"status": "healthy", "service": "momento-ws"}` when reachable. The frontend polls this endpoint to detect connectivity issues and show appropriate messaging to users.

**Minimal validation**: The endpoint only confirms the FastAPI application is running and responsive, without verifying model loading or database connectivity. This keeps the check fast and focused on basic reachability.

## Common Patterns

### Session Tracking

All endpoints support `session_id` metadata for tracking related operations. This enables correlating logs, analytics, and debugging across multiple WebSocket operations (transcribe → save → ask).

### Final Marker Protocol

Operations complete explicitly via an `is_final` marker rather than implicitly via connection closure. This design choice enables:

**Multiple operations per connection**: One WebSocket can handle sequential operations without reconnecting
**Clear boundaries**: Server knows when to trigger processing (save memory, generate answer)
**Resource cleanup**: Server can release operation-specific resources while keeping connection alive

### Error Handling

All endpoints follow consistent error handling:

- Exceptions during processing close the WebSocket with code 1011 (server error)
- Unsupported operations close with code 1003 (unsupported data)
- Normal completion uses code 1000
- Errors are logged server-side with full context

## Connection Lifecycle

Connections follow a simple lifecycle:

1. Client connects → server accepts
2. Client streams operation data
3. Server processes and streams responses
4. Client sends final marker → operation completes
5. Connection remains open for next operation or closes

This stateful approach (connection persists across operations) reflects WebSocket's design philosophy: maintain connection state rather than constantly reconnecting like HTTP.

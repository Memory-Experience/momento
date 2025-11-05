# API Package (Backend Server) Architecture

The API Package provides the backend server for the Momento intelligent memory system. It implements a FastAPI-based WebSocket server that handles real-time speech transcription, memory storage, and intelligent question answering using retrieval-augmented generation (RAG).

> **Note:** The name "API Package" stems from its role in defining, implementing, and exposing the application programming interface (API) using FastAPI for the Momento system, allowing clients (like the web frontend) to interact with its core functionalities.

## Overview

The API architecture is designed with modularity, scalability, and real-time performance in mind, enabling users to:

- **Transcribe audio in real-time**: Stream audio data and receive immediate transcription feedback using Faster Whisper
- **Store memories with semantic indexing**: Automatically chunk, embed, and index memories in a vector database for semantic search
- **Answer questions intelligently**: Retrieve relevant memories and generate contextual answers using local LLM models
- **Maintain clean separation of concerns**: Leverage dependency injection for testable, swappable components

## Architecture Components

### [WebSocket Endpoints](websocket_endpoints.md)

Three specialized WebSocket endpoints (`/ws/transcribe`, `/ws/memory`, `/ws/ask`) handle bidirectional streaming communication using Protocol Buffers for efficient data serialization.

### [AI Models](models.md)

Local AI models for speech-to-text (Faster Whisper), text embeddings (Qwen3-Embedding), and text generation (Qwen3-Instruct), all running without external API dependencies.

### [Core Services](services.md)

Modular services implementing vector storage (`VectorStoreService`), file persistence (`PersistenceService`), RAG processing (`LLMRAGService`), and relevance filtering (`ThresholdFilterService`) with clear interfaces and responsibilities. Services are assembled using container-based dependency injection for testability and flexibility.

## Architecture

```puml
@startuml
skinparam backgroundColor #FFFFFF
skinparam component {
  BackgroundColor #E8F5E9
  BorderColor #388E3C
  FontColor #1B5E20
}
skinparam interface {
  BackgroundColor #FFF3E0
  BorderColor #F57C00
  FontColor #E65100
}
skinparam database {
  BackgroundColor #E3F2FD
  BorderColor #1976D2
  FontColor #0D47A1
}

title "API Package - Component Architecture"

package "FastAPI WebSocket Server" {
  component [WebSocketHandler] as Handler

  interface "/ws/transcribe" as WSTranscribe
  interface "/ws/memory" as WSMemory
  interface "/ws/ask" as WSAsk
}

package "Service Layer" {
  component [TranscriptionServicer] as Transcriber
  component [MemoryPersistService] as Persist
  component [QuestionAnswerService] as QA
}

package "Core Services" {
  component [VectorStoreService] as VectorStore
  component [PersistenceService] as FileStore
  component [LLMRAGService] as RAG
  component [ThresholdFilterService] as Filter
}

package "Data Layer" {
  database "QDrant\nVector DB" as Qdrant
  database "JSON Files\n(recordings/)" as Files
}

package "AI Models" {
  component [FasterWhisper\nSTT Model] as Whisper
  component [Qwen3-Embedding\n0.6B] as Embedding
  component [Qwen3-Instruct\n1.7B] as LLM
}

package "Domain Models" {
  component [MemoryRequest] as Memory
  component [MemoryContext] as Context
}

skinparam linetype polyline

' WebSocket connections
WSTranscribe --> Handler
WSMemory --> Handler
WSAsk --> Handler

' Handler to servicers
Handler --> Transcriber
Handler --> Persist
Handler --> QA

' Servicer dependencies
Transcriber --> Whisper
Persist --> VectorStore
Persist --> FileStore
QA --> VectorStore
QA --> RAG
QA --> Filter

' Service to model connections
VectorStore --> Embedding
VectorStore --> Qdrant
FileStore --> Files
RAG --> LLM

' Domain model usage
VectorStore ..> Memory
VectorStore ..> Context
RAG ..> Memory
RAG ..> Context

@enduml
```

## Key Design Principles

### 1. Dependency Injection

All components are assembled through the `Container` class, which:

- Creates and configures all dependencies at startup
- Enables easy swapping of implementations (e.g., different embedding models)
- Improves testability by allowing mock injection
- Keeps configuration centralized

### 2. Repository Pattern

Data access is abstracted behind repository interfaces:

- `VectorStoreRepository` for QDrant operations
- `FileRepository` for JSON persistence
- Allows switching storage backends without changing business logic

### 3. Service Layer Architecture

Business logic is organized into focused services:

- Each service has a single, clear responsibility
- Services depend on abstractions, not concrete implementations
- Async/await patterns for non-blocking I/O operations

### 4. Domain-Driven Design

Core domain models (`MemoryRequest`, `MemoryContext`) represent business concepts:

- Rich domain objects with behavior, not just data containers
- Factory methods for object creation
- Encapsulation of domain logic

### 5. Streaming-First Design

All major operations support streaming:

- Audio transcription streams results as they're produced
- RAG responses stream token-by-token for better UX
- WebSocket protocol enables bidirectional streaming

## Data Flows

The system implements two primary data flows for handling memory operations.

### Memory Storage Flow

```
Client sends audio/text → Buffer chunks → Create MemoryRequest → Save to JSON
→ Chunk text → Generate embeddings → Store in QDrant → Return memory ID
```

### Question Answering Flow

```
Client sends question → Buffer chunks → Generate embedding → Search QDrant
→ Filter by threshold → Stream contexts → Generate answer (RAG) → Stream tokens
```

## Technology Stack

- **Web Framework**: FastAPI with WebSocket support
- **Protocol**: Protocol Buffers (protobuf) for message serialization
- **STT Model**: Faster Whisper (local inference)
- **Embedding Model**: Qwen3-Embedding-0.6B (GGUF quantized)
- **LLM Model**: Qwen3-1.7B-Instruct (GGUF quantized)
- **Vector Database**: QDrant (in-memory mode)
- **Model Runtime**: llama.cpp for GGUF model inference
- **Text Processing**: spaCy for sentence chunking

## Scalability Considerations

The current architecture is designed as a single-instance deployment suitable for personal use with 1-10 concurrent users. The in-memory vector store provides fast access but naturally limits the total memory count based on available RAM. Local model inference enables complete offline operation without external API dependencies.

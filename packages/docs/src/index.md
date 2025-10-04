# Momento - Intelligent Memory Experience System

Momento is a sophisticated intelligent memory storage system with retrieval capabilities. It combines advanced AI technologies including speech-to-text for dictation, vector embeddings, and retrieval-augmented generation (RAG) to create an intelligent memory assistant that can capture, store, and answer questions about your memories and experiences.

## Overview

Momento transforms spoken or written content into searchable memories, enabling users to:

- **Record and transcribe** audio in real-time using advanced speech recognition
- **Store memories** with intelligent vector-based indexing for semantic search
- **Query memories** using natural language questions with AI-powered responses
- **Interact seamlessly** through a modern web interface with real-time WebSocket communication

## High-Level Architecture

```puml
@startuml
skinparam backgroundColor #FFFFFF
skinparam component {
  BackgroundColor #E3F2FD
  BorderColor #1976D2
  FontColor #1976D2
}
skinparam database {
  BackgroundColor #FFF3E0
  BorderColor #F57C00
  FontColor #F57C00
}

title "Momento - High-Level Architecture"

component [Web Frontend] as Frontend
component [Backend] as Backend
package "Backend" as Backend {
    component FastAPI as Api
    component [AI Models\n(Whisper, Qwen3, BERT)] as AIModels
    database "QDrant Vector DB" as QDrant
    database "File Storage" as FileStorage
}

' Three main WebSocket endpoints
Frontend --> Api : /ws/transcribe\n(STT)
Frontend --> Api : /ws/memory\n(Store Memories)
Frontend --> Api : /ws/ask\n(RAG Q&A)

' Backend processes
Api --> AIModels : STT\nText Generation\nVector Embeddings
Api --> QDrant : Semantic Search\nVector Storage
Api --> FileStorage : Metadata Persistence

' RAG flow
QDrant --> Api : Relevant Memories
AIModels --> Api : Generated\nResponses

note top of QDrant
  **Vector Database**
  • Memory indexing
  • Semantic similarity search
end note

note top of Frontend
  **WebSocket Communication**
  • Real-time bidirectional communication
  • Binary transmission using protobuf
end note

note bottom of AIModels
  **RAG Pipeline**
  Query → Search → Context → Generate → Stream
end note

@enduml
```

## Key Features

### Audio & Text Input

- Real-time speech-to-text transcription
- Direct text input support
- WebSocket-based streaming communication

### Memory Storage

- Automatic text chunking and vector embedding generation
- Semantic indexing in QDrant vector database
- File-based persistence for audio and metadata

### Intelligent Retrieval

- Natural language question processing
- Semantic similarity search across stored memories
- Context filtering by relevance thresholds

### AI-Powered Responses

- RAG-based answer generation using Qwen3 LLM
- Streaming response delivery
- Context-aware answers based on retrieved memories

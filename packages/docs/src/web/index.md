# Web Frontend Architecture

The Momento web frontend is a Next.js application that provides a real-time chat interface for memory storage and retrieval. It communicates with the FastAPI backend via WebSocket connections using Protocol Buffers for efficient data transmission.

## Architecture Overview

### Application Structure

```
├── app
│   ├── globals.css                 # Global styles
│   ├── layout.tsx                  # Root layout with theme provider
│   └── page.tsx                    # Main page containing Chat component
├── components
│   ├── Chat.tsx                    # Main chat interface component
│   ├── controls                    # Input/recording controls
│   │   ├── AudioRecorder.tsx
│   │   └── TranscribedRecorder.tsx
│   ├── ThemeProvider.tsx
│   └── ui                          # Reusable UI components
│       ├── Header.tsx
│       ├── Message.tsx
│       ├── MessageEmptyState.tsx
│       └── MessageList.tsx
├── context
│   └── RecordingContext.tsx        # Audio recording state
├── hooks
│   └── useWebSocket.tsx            # WebSocket connection management
├── types
│   ├── Message.ts                  # Message type definitions
│   └── RecordingContextType.ts     # Recording state types
└── utils
    └── message.reducers.ts         # Reducers for real-time messages
```

### Core Components

#### Chat Component

The main orchestrator that handles:

- Text input via Textarea
- Mode switching between "memory" and "question"
- WebSocket connection management for `/ws/memory` and `/ws/ask` endpoints
- Message state management and display
- Integration with transcription services via `TranscribedRecorder.tsx`

#### WebSocket Hook (`useWebSocket`)

Manages websocket connections with the backend:

- Connection lifecycle management
- Event listener management
- Binary protobuf message handling
- Graceful socket cleanup and closure of sockets using `ChunkMetadata.is_final`

#### Controls

- **AudioRecorder**: Handles microphone input and audio data capture
- **TranscribedRecorder**: Combines audio recording with real-time transcription

## WebSocket Communication

The frontend communicates with the backend through three specialized WebSocket endpoints, each serving a distinct purpose:

### Protocol Definition

The data is send in binary using [Protocol Buffers](https://protobuf.dev/). There are various message types in the `stt.proto` specification:

```protobuf
enum ChunkType {
  MEMORY = 0;      // Memory messages
  QUESTION = 1;    // Question/query messages
  TRANSCRIPT = 2;  // Transcription messages (audio or text)
  ANSWER = 3;      // Generated responses
}
```

For the full definition see [`stt.proto`](https://github.com/Memory-Experience/momento/blob/main/packages/protos/stt.proto)

### WebSocket Endpoints

#### 1. `/ws/transcribe` - Real-time Speech-to-Text

- **Purpose**: Live audio transcription
- **Flow**:
    - Send audio chunks → Receive transcribed text segments
    - Used by `TranscribedRecorder` component
- **Message Types**: `ChunkType.TRANSCRIPT`

#### 2. `/ws/memory` - Memory Storage

- **Purpose**: Save memories (text/audio) to the system
- **Flow**:
    - Send memory data → Receive confirmation with memory ID
    - Triggers vector indexing in backend
- **Message Types**: `ChunkType.MEMORY`

#### 3. `/ws/ask` - RAG Question Answering

- **Purpose**: Query stored memories using natural language
- **Flow**:
    - Send question → Receive relevant memories → Receive generated answer
    - Implements retrieval-augmented generation (RAG)
- **Message Types**: `ChunkType.QUESTION` → `ChunkType.MEMORY`/`ChunkType.ANSWER`

Websockets connections are implemented and managed using the custom `useWebSocket` hook.

Key features:

- **Binary Protocol**: All messages use Protocol Buffers encoding
- **Graceful Disconnection**: Sends final marker before closing
- **Event Management**: Proper cleanup of listeners
- **Connection State**: Tracks connection status

### Message Flow Examples

#### Memory Storage Flow

```typescript
// 1. User types text and clicks "Save Memory"
const memoryChunk: MemoryChunk = {
    textData: "Had lunch at the park today",
    metadata: {
        memoryId: crypto.randomUUID(),
        sessionId: "",
        type: ChunkType.MEMORY,
        isFinal: true,
        score: 0,
    },
};

// 2. Send to /ws/memory endpoint
send(MemoryChunk.encode(memoryChunk).finish());

// 3. Receive confirmation
// Response: "Memory saved with ID: uuid-string"
```

#### Question Answering Flow

```typescript
// 1. User asks question
const questionChunk: MemoryChunk = {
    textData: "What did I have for lunch?",
    metadata: {
        sessionId: crypto.randomUUID(),
        type: ChunkType.QUESTION,
        isFinal: true,
        score: 0,
    },
};

// 2. Send to /ws/ask endpoint
send(MemoryChunk.encode(questionChunk).finish());

// 3. Receive relevant memories (MEMORY chunks)
// 4. Receive streaming answer (ANSWER chunks)
```

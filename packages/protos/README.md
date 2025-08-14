# Protocol Buffers Package

Protocol Buffer definitions and code generation for the speech transcription gRPC service. Generates Python bindings for the API server and TypeScript definitions for web clients.

## Prerequisites

- **Node.js** v22
- **Python** 3.12
- **pnpm** v10.14+ for Node.js package management
- **uv** v0.8+ for Python package management

> **Note**: For installation instructions, refer to the [root README](../../README.md#prerequisites).

## Getting Started

### 1. Install Dependencies

Install all project dependencies from the root directory:

```bash
uv sync          # Python dependencies
pnpm install     # Node.js dependencies
```

### 2. Generate Protocol Buffer Code

Generate all required code from the protos package:

```bash
cd packages/protos
pnpm run build
```

This command executes the following steps:

1. **`generate:py`** - Generates Python protobuf code
2. **`generate:py:fix`** - Fixes import paths in generated Python code
3. **`generate:ts`** - Generates TypeScript protobuf definitions
4. **`generate:ts:clients`** - Generates TypeScript gRPC-Web clients

## Protocol Buffer Definitions

- `stt.proto`: Defines a bidirectional streaming service for real-time speech transcription.

## Development

### Managing Dependencies

- **Python**: Use `uv add` to manage Python dependencies
- **Node.js**: Use `pnpm add` or `pnpm install` to manage Node.js packages

See [`pyproject.toml`](pyproject.toml) and [`package.json`](package.json) for complete dependency lists.

### Available Scripts

- `pnpm run build` - Clean output directory and generate all Protocol Buffer code
- `pnpm run generate` - Generate protocol buffer code
- `pnpm run clean` - Remove generated code

For all available scripts, see [`package.json`](package.json).

### Generated Code

Generated code is excluded from version control and built automatically by the CI/CD pipeline. The generated files include:

- **Python**: `generated/py/stt_pb2.py`, `generated/py/stt_pb2_grpc.py`
- **TypeScript**: `generated/ts/stt.ts`, `generated/ts/clients/stt.ts`

## Usage Examples

### Python (API Server)

```python
from protos.generated.py import stt_pb2, stt_pb2_grpc

# Create a gRPC stub
stub = stt_pb2_grpc.TranscriptionServiceStub(channel)

# Create message instances
audio_chunk = stt_pb2.AudioChunk(data=audio_bytes)
response = stt_pb2.StreamResponse(
    transcript=stt_pb2.Transcript(text="Hello world")
)
```

### TypeScript (Web Client)

```typescript
// Import generated types
import { AudioChunk, StreamResponse } from "./generated/ts/stt";

// Import gRPC-Web client
import { TranscriptionServiceClient } from "./generated/ts/clients/stt";

// Create client and message
const client = new TranscriptionServiceClient("http://localhost:8080");
const audioChunk = AudioChunk.create({ data: new Uint8Array(audioData) });
```

# Protocol Buffers Package

Protocol Buffer definitions and code generation for the speech transcription gRPC service. Generates Python bindings for the API server and TypeScript definitions for web clients.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](../../README.md#prerequisites) first and follow the steps to install dependencies in the [root README Quick Start section](../../README.md#quick-start).

### Generate Protocol Buffer Code

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

**Development & Build:**
```bash
pnpm run build        # Run clean and generate
pnpm run generate     # Generate protocol buffer code
pnpm run clean        # Remove generated code
```

**Code Quality:**
```bash
# From project root (recommended)
pnpm run format       # Format all files
pnpm run format:check # Check formatting
pnpm run lint         # Run all linting

# Alternative workspace commands
pnpm run -w format    # Format from any package directory

# Python-specific tools (alternative)
uvx ruff format       # Format Python code
uvx ruff check        # Lint Python code
uvx ruff check --fix  # Lint and automatically fix issues
```

For complete script details, see [`package.json`](package.json).

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

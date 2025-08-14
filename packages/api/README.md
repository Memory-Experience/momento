# API Package - gRPC Speech Transcription Server

A Python-based gRPC server that provides real-time speech transcription services using Faster Whisper and Protocol Buffers for efficient communication.

## Prerequisites

- **Python** 3.12
- **uv** v0.8+ for Python package management
- **FFmpeg** required for audio processing

**Note**: For installation instructions, refer to the [root README](../../README.md#prerequisites).

## Getting Started

### 1. Install Dependencies

Install project dependencies from the root directory:

```bash
# From the project root
uv sync
```

### 2. Generate Protocol Buffer Code

The API requires generated Protocol Buffer code to communicate with clients. From the [`protos`](../protos/) package:

```bash
pnpm run build
```

Or from the root directory:

```bash
pnpm run --dir packages/protos build
```

### 3. Start the Server

```bash
# From the api package directory
cd packages/api
pnpm run start

# Alternative: run directly with uv
uv run main.py
```

The gRPC server will start on `localhost:50051` ready to accept requests.

## Development

### Managing Dependencies

Use `uv add` to add new Python dependencies, which automatically updates both the local [`pyproject.toml`](pyproject.toml) and the workspace [`pyproject.toml`](../../pyproject.toml):

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Available Scripts

- `pnpm run start` - Start the gRPC server
- `uvx ruff format` - Format Python code
- `uvx ruff check` - Lint Python code
- `uvx ruff check --fix` - Lint and automatically fix issues

### Code Quality

This package uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is inherited from the root workspace configuration.

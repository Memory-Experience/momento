# API Package

A Python-based gRPC server providing real-time speech transcription services using Faster Whisper and Protocol Buffers for efficient communication.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](../../README.md#prerequisites) and follow the steps to install dependencies and protocol buffer definitions in the [root README Quick Start section](../../README.md#quick-start).

### Start the Server

```bash
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

**Server Management:**
```bash
pnpm run start        # Start the gRPC server
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

### Code Quality

This package uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is inherited from the root workspace configuration.

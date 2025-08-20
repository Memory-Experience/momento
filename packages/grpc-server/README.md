# gRPC server Package

A Python-based gRPC server providing real-time speech transcription services using Faster Whisper and Protocol Buffers for efficient communication.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](../../README.md#prerequisites) and follow the steps to install dependencies and protocol buffer definitions in the [root README Quick Start section](../../README.md#quick-start).

### Configuration

The server requires database connection configuration via environment variables. You can provide these in several ways:

#### Environment Variables

The following environment variables are required:

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `uzh` |
| `DB_USER` | Database username | `uzh` |
| `DB_PASSWORD` | Database password | `password` |

```bash
export DB_PASSWORD=password
DB_HOST=localhost DB_PORT=5432 DB_NAME=uzh DB_USER=uzh pnpm run start
```

#### Configuration Files

You can use `.env` files to manage configuration:

**Option 1: Default configuration**
Create a `.env` file in the `grpc-server` package with your database settings:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=uzh
DB_USER=uzh
DB_PASSWORD=password
```

**Option 2: Environment-specific configuration**
Use the `-e` flag to specify a custom environment file:

```bash
# Use a specific .env file
uv run main.py -e .env.local
uv run main.py -e .env.production
```

### Start the Server

```bash
cd packages/grpc-server
pnpm run dev          # Uses .env.local configuration
pnpm run start        # Uses default .env or system environment variables

# Alternative: run directly with uv
uv run main.py                    # Default configuration
uv run main.py -e .env.local      # Custom environment file
uv run main.py -v                 # Enable verbose/debug logging
uv run main.py -e .env.local -v   # Custom env file + verbose logging
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
pnpm run dev          # Start the gRPC server with .env.local configuration
pnpm run start        # Start the gRPC server with default configuration
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

### Command Line Options

The server supports the following command line arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `-e <file>` | Load environment variables from specified file | `-e .env.production` |
| `-v` | Enable verbose/debug logging | `-v` |

**Examples:**
```bash
# Development with debug logging
uv run main.py -e .env.local -v
```

### Code Quality

This package uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is inherited from the root workspace configuration.
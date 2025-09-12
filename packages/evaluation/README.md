# Evaluation Package

A python based evaluation runner providing a Dataset interface with which the runner can evaluate the backend API package.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](https://github.com/Memory-Experience/momento/blob/main/README.md#prerequisites) and follow the steps to install dependencies and protocol buffer definitions in the [root README Quick Start section](https://github.com/Memory-Experience/momento/blob/main/README.md#quick-start).

### Start the Server

```bash
cd packages/evaluation
pnpm run start

# Alternative: run directly with uv
uv run main.py
```

## Development

### Managing Dependencies

Use `uv add` to add new Python dependencies, which automatically updates both the local [`pyproject.toml`](https://github.com/Memory-Experience/momento/blob/main/packages/evaluation/pyproject.toml) and the workspace [`pyproject.toml`](https://github.com/Memory-Experience/momento/blob/main/pyproject.toml):

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

**Testing:**

```bash
pnpm run test         # Run pytest with uv (Python 3.12)
pnpm run test:ci      # Quiet test run for CI

# Filter tests
pnpm run test -- -k dataset

# Run a single file
pnpm run test -- tests/test_dataset.py -q
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

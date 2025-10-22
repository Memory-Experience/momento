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

### Running the jupyter notebook

Here we show how to create a jupyter kernel in uv. For further information see the official uv documentation on [using uv with Jupyter](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project).

```bash
uv sync --dev

uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=momento

# If you want to start a jupyter server instead of using it directly in vscode
uv run --with jupyter jupyter lab
```

### Loading Result Files

Large JSON result files in `evaluation/runs` are stored using [Git LFS](https://git-lfs.com/) (Large File Storage) due to GitHub's 100MB file size limit.

**Install Git LFS:**

- **macOS**: `brew install git-lfs`
- **Ubuntu/Debian**: `sudo apt-get install git-lfs`
- **Windows**: Download from [git-lfs.com](https://git-lfs.com/)

After installation, initialize Git LFS and pull the files:

```bash
git lfs install
git lfs pull
```

The JSON files in `evaluation/runs` will then be available for analysis.

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

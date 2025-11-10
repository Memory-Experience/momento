# UZH Master Project &middot; [![Build](https://github.com/Memory-Experience/momento/actions/workflows/build.yaml/badge.svg)](https://github.com/Memory-Experience/momento/actions/workflows/build.yaml) &middot; [![Pages](https://github.com/Memory-Experience/momento/actions/workflows/deploy.yaml/badge.svg)](https://memory-experience.github.io/momento/)

**Momento** is a memory prosthesis system that transforms spoken or written content into searchable memories. It captures conversations, meetings, or personal notes through real-time speech-to-text transcription, then organizes this information into a semantic memory system that you can query naturally using everyday language.

Think of it as an external memory that remembers everything you tell it and helps you recall information when you need it.

## Architecture

This monorepo contains multiple packages:

- **[`packages/api/`](https://github.com/Memory-Experience/momento/blob/main/packages/api/README.md)** - Backend Server (FastAPI + WebSocket) for speech transcription and memory retrieval
- **[`packages/protos/`](https://github.com/Memory-Experience/momento/blob/main/packages/protos/README.md)** - Protocol Buffers definitions and code generation
- **[`packages/web/`](https://github.com/Memory-Experience/momento/blob/main/packages/web/README.md)** - Frontend with real-time audio recording and transcription
- **[`packages/evaluation/`](https://github.com/Memory-Experience/momento/blob/main/packages/evaluation/README.md)** - Benchmarking suite evaluating RAG performance
- **[`packages/docs/`](https://github.com/Memory-Experience/momento/blob/main/packages/docs/README.md)** - MkDocs Documentation site of the project

### Why This Structure?

**Monorepo with `packages/`**: We use the standard monorepo convention where each package is a self-contained module with its own dependencies and build process. This is the established pattern for monorepo workspaces. Each package is an independent module and may have its own `src/` directory internally depending on the used technology and best practices thereof.

**Documentation as a Package**: The `packages/docs/` location might seem unusual, but it's intentional. Our documentation is a deployable package with its own `package.json`, dependencies, and build process. It generates API documentation from the codebase and deploys independently to GitHub Pages. Placing it at the root would break the workspace structure and complicate dependency management.

For detailed architecture information, see:

- [API Package (Backend Server) Architecture](https://memory-experience.github.io/momento/api/architecture/)
- [Web Package Architecture](https://memory-experience.github.io/momento/web/architecture/)
- [Evaluation Package Architecture](https://memory-experience.github.io/momento/evaluation/architecture/)

## Prerequisites

### System Requirements

- **Node.js** v22 (see [`.nvmrc`](https://github.com/Memory-Experience/momento/blob/main/.nvmrc))
- **Python** 3.12 (see [`.python-version`](https://github.com/Memory-Experience/momento/blob/main/.python-version))
- **pnpm** v10.14+ for Node.js package management (see [installation instructions using NVM (recommended)](https://github.com/Memory-Experience/momento/blob/main/README.md#1-install-nodejs-and-pnpm))
- **uv** v0.8+ for Python package management (see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **FFmpeg** required by the transcription API

### Installation

#### 1. Install Node.js and pnpm

Install [Node Version Manager (NVM)](https://github.com/nvm-sh/nvm#installing-and-updating), then run:

```bash
# Install and use the required Node.js version
nvm install
nvm use

# Install pnpm globally
npm install -g pnpm@10.14.0
```

You may also refer to the [pnpm installation guide](https://pnpm.io/installation#using-npm) for more information.

#### 2. Install uv and Python

Follow the official installation instruction for [uv](https://docs.astral.sh/uv/getting-started/installation/), then install Python using uv's built-in Python installer:

```bash
uv python install
```

#### 3. Install FFmpeg

Follow the [FFmpeg download and installation](https://ffmpeg.org/download.html) instructions for your system.

## Quick Start

Follow these steps to get the application running:

### 1. Install Dependencies

```bash
# Install all dependencies from the project root
pnpm install  # Node.js dependencies
uv sync       # Python dependencies
```

### 2. Generate Protocol Buffers

```bash
cd packages/protos
pnpm run build
```

### 3. Start the Application

Start the API server (Backend Server):

```bash
cd packages/api
pnpm run start
```

![API package start command](https://github.com/Memory-Experience/momento/blob/main/docs/images/api_start_command.svg)

In a new terminal, start the web frontend:

```bash
cd packages/web
pnpm run dev
```

The web interface will be accessible at `http://localhost:3000` (or the URL shown in the terminal output).
![web package dev command](https://github.com/Memory-Experience/momento/blob/main/docs/images/web_dev_command.svg)
![web UI after starting dev server](https://github.com/Memory-Experience/momento/blob/main/docs/images/momento_web_ui.png)

## Development

This project uses `pnpm` for Node.js packages and `uv` for Python packages. For consistency, Python packages include a `package.json` with scripts that invoke `uv` commands, allowing uniform use of `pnpm run` across the entire monorepo.

> **Note**: Warnings about "Local package.json exists, but node_modules missing" in Python packages are expected and can be ignored. Python packages use `package.json` only for script orchestration via pnpm (invoking `uv` commands), allowing uniform use of `pnpm run` commands across the entire monorepo. As Python packages, they don't have Node.js dependencies, hence no `node_modules` folder. Dependencies for Python packages are managed exclusively via `uv` and `pyproject.toml`.

### Package Management

- **[pnpm](https://pnpm.io/installation)**: JavaScript/TypeScript monorepo management
- **[uv](https://docs.astral.sh/uv/)**: Fast Python package manager

### Code Quality Tools

- **Python**: [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- **JavaScript/TypeScript**: [ESLint](https://eslint.org/) and [Prettier](https://prettier.io/)
- **Git Hooks**: [Husky](https://typicode.github.io/husky/) with [lint-staged](https://github.com/lint-staged/lint-staged)

Python tools can be run using `uvx ruff` (similar to `npx` for Node.js).

**Configuration files:**

- [`.lintstagedrc.yaml`](https://github.com/Memory-Experience/momento/blob/main/.lintstagedrc.yaml) - Pre-commit hooks
- [`pyproject.toml`](https://github.com/Memory-Experience/momento/blob/main/pyproject.toml) - Python workspace
- [`ruff.toml`](https://github.com/Memory-Experience/momento/blob/main/ruff.toml) - Ruff configuration
- [`pnpm-workspace.yaml`](https://github.com/Memory-Experience/momento/blob/main/pnpm-workspace.yaml) - Node.js workspace configuration

### Available Scripts

```bash
pnpm run format       # Format all files
pnpm run format:check # Check formatting
pnpm run lint         # Run all linting
pnpm run test         # Run tests across all packages
```

When running tests, results are displayed in the terminal showing test pass/fail status and coverage reports.

## Documentation

Full documentation is available at: https://memory-experience.github.io/momento/ deployed from the [`packages/docs/`](https://github.com/Memory-Experience/momento/tree/main/packages/docs) package.

## Troubleshooting

### Husky Git Hooks on Windows/WSL

When using Node.js version managers (like NVM) on Windows or WSL, you may encounter the following error during Git commits:

```
node_modules/.bin/lint-staged: 20: exec: node: not found
husky - pre-commit script failed (code 127)
husky - command not found in PATH=...
```

This occurs because Git hooks run in a minimal environment that doesn't load your shell configuration where the version manager sets up Node.js in the PATH.
As per the [official documentation](https://typicode.github.io/husky/how-to.html#solution):

**Solution:**

Create or edit `~/.config/husky/init.sh` to initialize your Node.js version manager:

```bash
# ~/.config/husky/init.sh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
```

Husky automatically sources this file before each Git hook, ensuring Node.js is available in the hook environment.

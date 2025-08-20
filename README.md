# UZH Master Project &middot; [![Build](https://github.com/oberpierre/uzh-masterproject/actions/workflows/build.yaml/badge.svg)](https://github.com/oberpierre/uzh-masterproject/actions/workflows/build.yaml)

A real-time speech-to-text transcription system built with gRPC, featuring a Python backend API and Next.js web frontend.

## Architecture

This monorepo contains three main packages:

- **[`packages/api/`](packages/api/README.md)** - Python gRPC server for speech transcription
- **[`packages/protos/`](packages/protos/README.md)** - Protocol Buffers definitions and code generation
- **[`packages/web/`](packages/web/README.md)** - Next.js frontend with real-time audio recording and transcription

## Prerequisites

### System Requirements

- **Node.js** v22 (see [`.nvmrc`](.nvmrc))
- **Python** 3.12 (see [`.python-version`](.python-version))
- **pnpm** v10.14+ for Node.js package management
- **uv** v0.8+ for Python package management
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

#### 2. Install uv and Python

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install Python:

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

Start the API server:

```bash
cd packages/api
pnpm run dev
```

In a new terminal, start the web frontend:

```bash
cd packages/web
pnpm run dev
```

The web application will be available at `http://localhost:3000`.

## Development

This project uses `pnpm` for Node.js packages and `uv` for Python packages. For consistency, Python packages include a `package.json` with scripts that invoke `uv` commands, allowing uniform use of `pnpm run` across the entire monorepo.

**Note**: Warnings about "Local package.json exists, but node_modules missing" in Python packages are expected and can be ignored.

### Code Quality Tools

- **Python**: [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- **JavaScript/TypeScript**: [ESLint](https://eslint.org/) and [Prettier](https://prettier.io/)
- **Git Hooks**: [Husky](https://typicode.github.io/husky/) with [lint-staged](https://github.com/lint-staged/lint-staged)

Python tools can be run using `uvx ruff` (similar to `npx` for Node.js).

**Configuration files:**

- [`.lintstagedrc.yaml`](.lintstagedrc.yaml) - Pre-commit hooks
- [`pyproject.toml`](pyproject.toml) - Python workspace
- [`ruff.toml`](ruff.toml) - Ruff configuration
- [`pnpm-workspace.yaml`](pnpm-workspace.yaml) - Node.js workspace configuration

### Available Scripts

```bash
pnpm run format       # Format all files
pnpm run format:check # Check formatting
pnpm run lint         # Run all linting
```

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

# Web Frontend - Real-time Audio Recording Interface

A Next.js application that provides a modern web interface for real-time audio recording and transcription. Built with TypeScript, Tailwind CSS, and WebRTC audio capture.

## Prerequisites

- **Node.js** v22+
- **pnpm** v10.14+
- Modern web browser with WebRTC support

> **Note**: For installation instructions, refer to the [root README](../../README.md#prerequisites).

## Getting Started

### 1. Install Dependencies

Install all project dependencies from the root directory:

```bash
pnpm install
```

### 2. Start Development Server

From the web package directory:

```bash
cd packages/web
pnpm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Production Build

```bash
pnpm run build
pnpm run start
```

## Development

### Managing Dependencies

Use `pnpm add` or `pnpm install` to manage Node.js package dependencies.

See [`package.json`](package.json) for a complete list of dependencies.

### Available Scripts

- `pnpm run dev` - Start development server with Turbopack
- `pnpm run build` - Build for production
- `pnpm run start` - Start production server
- `pnpm run lint` - Run ESLint

Global formatting scripts from the root workspace can be run using:
```bash
# From the project root
pnpm format          # Format all files
pnpm format:check    # Check formatting

# Or from this package directory
pnpm run -w format
```

### Code Quality

This package uses ESLint and Prettier with automatic formatting via git hooks. Configuration is inherited from the [root workspace](../../.lintstagedrc.yaml).

## Attribution

This frontend package is based on the [HumeAI EVI Next.js starter template](https://github.com/HumeAI/hume-evi-next-js-starter), available via [Vercel](https://vercel.com/templates/ai/empathic-voice-interface-starter). The original HumeAI functionality has been removed and replaced with our speech transcription implementation, while retaining the base UI components and project structure.

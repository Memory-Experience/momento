# Web Frontend

A Next.js application providing a modern web interface for real-time audio recording and transcription. Built with TypeScript, Tailwind CSS, and WebRTC audio capture.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](../../README.md#prerequisites) and follow the steps to install dependencies and protocol buffer definitions in the [root README Quick Start section](../../README.md#quick-start).

### Start Development Server

```bash
cd packages/web
pnpm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
pnpm run build
pnpm run start
```

## Development

### Managing Dependencies

Use `pnpm add` or `pnpm install` to manage Node.js package dependencies.

See [`package.json`](package.json) for a complete list of dependencies.

### Available Scripts

**Development & Build:**
```bash
pnpm run dev          # Start development server with Turbopack
pnpm run build        # Build for production
pnpm run start        # Start production server
pnpm run lint         # Run ESLint
```

**Code Quality:**
```bash
# From project root (recommended)
pnpm run format       # Format all files
pnpm run format:check # Check formatting
pnpm run lint         # Run all linting

# Alternative workspace commands
pnpm run -w format    # Format from any package directory
```

### Code Quality

This package uses ESLint and Prettier with automatic formatting via git hooks. Configuration is inherited from the [root workspace](../../.lintstagedrc.yaml).

## Attribution

This frontend package is based on the [HumeAI EVI Next.js starter template](https://github.com/HumeAI/hume-evi-next-js-starter), available via [Vercel](https://vercel.com/templates/ai/empathic-voice-interface-starter). The original HumeAI functionality has been removed and replaced with our speech transcription implementation, while retaining the base UI components and project structure.

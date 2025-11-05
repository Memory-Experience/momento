# Web Frontend

A [Next.js](https://nextjs.org) application providing a real-time interface for memory storage and question-answering capabilities through WebSocket-based communication with the backend API.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](https://github.com/Memory-Experience/momento/blob/main/README.md#prerequisites) and follow the steps to install dependencies and protocol buffer definitions in the [root README Quick Start section](https://github.com/Memory-Experience/momento/blob/main/README.md#quick-start).

### Start the Development Server

```bash
cd packages/web
pnpm run dev
```

The development server will start and display a URL in the terminal (typically `http://localhost:3000`). Open this URL in your browser to see the application. The page auto-updates as you edit files.

![web package dev command](https://github.com/Memory-Experience/momento/blob/main/docs/images/web_dev_command.svg)
![web UI after starting dev server](https://github.com/Memory-Experience/momento/blob/main/docs/images/momento_web_ui.png)

### Build for Production

```bash
pnpm run build    # Create optimized production build
pnpm run start    # Start production server
```

The production server will start and display the URL in the terminal where the application is accessible.

## Development

### Managing Dependencies

Use `pnpm` to add new dependencies, which automatically updates the [`package.json`](https://github.com/Memory-Experience/momento/blob/main/packages/web/package.json):

```bash
# Add a new dependency
pnpm add package-name

# Add a development dependency
pnpm add -D package-name
```

### Available Scripts

**Development:**

```bash
pnpm run dev          # Start development server (displays URL in terminal)
pnpm run build        # Create optimized production build
pnpm run start        # Start production server (displays URL in terminal)
```

**Testing:**

```bash
pnpm run test         # Run all tests once with coverage
pnpm run test:ci      # Run all tests in silent mode (CI/CD)
pnpm run test:watch   # Run tests in watch mode (re-runs on file changes)

# Run specific test file
pnpm run test -- Chat.test.tsx
# Run tests matching pattern
pnpm run test -- --testNamePattern="streaming"
```

Tests execute and display results in the terminal, showing:

- Test suite pass/fail status
- Individual test results
- Coverage percentages for statements, branches, functions, and lines
- Detailed coverage report available in `coverage/lcov-report/index.html`

**Code Quality:**

```bash
# Linting
pnpm run lint         # Run all linting (ESLint)

# Formatting from project root (recommended)
pnpm run format       # Format all files
pnpm run format:check # Check formatting without changes
```

**Note:** The formatting uses the workspace-level Prettier and Ruff configurations and should be run from the project root.

## Testing

Tests use [Jest](https://jestjs.io/) and [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) following user-centric testing principles.

### Test Structure

Tests are located alongside their components with the `.test.tsx` extension:

```
components/
  Chat.tsx
  Chat.test.tsx          # Tests for Chat component
  controls/
    AudioRecorder.tsx
    AudioRecorder.test.tsx
```

### Test Coverage

Coverage is automatically collected when running tests (see [`jest.config.ts`](https://github.com/Memory-Experience/momento/blob/main/packages/web/jest.config.ts) for configuration details). The setup uses the v8 provider and includes all source files while excluding build artifacts, configuration files, and type definitions.

**Coverage Reports:**

After running tests, coverage reports are available in multiple formats:

- **Terminal**: Summary statistics displayed after test completion
- **HTML**: Open `coverage/lcov-report/index.html` in your browser for an interactive, detailed view
- **LCOV**: `coverage/lcov.info` for CI/CD integration

**Note**: The coverage may be inaccurate if only a subset of tests are run (such as with the `pnpm run test -- <test-name>`). If you need accurate coverage, run all tests with `pnpm run test` or `pnpm run test:ci`.

### Code Quality

This package uses:

- **[ESLint](https://eslint.org/)** for JavaScript/TypeScript linting (see `eslint.config.mjs`)
- **[Prettier](https://prettier.io/)** for code formatting (via root workspace configuration)
- **[TypeScript](https://www.typescriptlang.org/)** for type safety (see `tsconfig.json`)

Configuration follows Next.js conventions and is integrated with the workspace-level tooling.

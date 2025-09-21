# Documentation Package

A documentation site for the UZH Masterproject, built with MkDocs for Python API docs and Typedoc for TypeScript API docs. This package collects and publishes documentation for all submodules in the monorepo.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](https://github.com/Memory-Experience/momento/blob/main/README.md#prerequisites) and follow the steps to install dependencies and protocol buffer definitions in the [root README Quick Start section](https://github.com/Memory-Experience/momento/blob/main/README.md#quick-start).

### Available Scripts

**Build documentation:**

- `pnpm run build` — Builds both TypeScript and Python documentation

**Serve documentation:**

- `pnpm run start` — Serves the documentation locally

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Output

- The generated documentation site will be available in the `site/` directory after building.

## Structure

- **MkDocs**: Python API documentation
- **Typedoc**: TypeScript API documentation

## Deployment

Documentation is automatically built and deployed to GitHub Pages via CI/CD on pushes to the `main` branch.

# Qdrant Package

A vector database service running Qdrant through Apptainer for efficient similarity search and vector storage.

## Getting Started

**Prerequisites**: Complete the setup steps in the [root README Prerequisites section](https://github.com/Memory-Experience/momento/blob/main/README.md#prerequisites).

### Build the Container

Build the Qdrant Apptainer image from Docker Hub:

```bash
cd packages/qdrant
pnpm run build
```

This will create a `qdrant.sif` Apptainer image file based on Qdrant v1.7.3.

### Start the Server

```bash
cd packages/qdrant
pnpm run start
```

The Qdrant server will start with:

- Storage bound to `./storage`
- Snapshots bound to `./snapshots`
- Configuration from `./config/qdrant.yaml`
- Logs written to `./logs`
- CPU cores 0-3 (configurable via `taskset`)

The server runs on the default Qdrant ports:

- HTTP API: `http://localhost:6333`
- gRPC API: `localhost:6334`

## Configuration

### Qdrant Configuration

The main configuration file is located at [`config/qdrant.yaml`](config/qdrant.yaml). Key settings include:

- **Logging**: Set to `INFO` level
- **Telemetry**: Disabled for privacy
- **Storage**: Persistent storage and snapshots paths
- **On-disk payload**: Enabled to reduce memory usage
- **WAL settings**: Configured for optimal write performance
- **Optimizers**: Background merges disabled during ingest to avoid RAM spikes

Refer to the [Qdrant documentation](https://qdrant.tech/documentation/guides/configuration/) for more configuration options.

### CPU Affinity

The `taskset -c 0-3` command in the start script pins Qdrant to CPU cores 0-3. Adjust this based on your system's available cores and workload requirements.

## Development

### Available Scripts

**Container Management:**

```bash
pnpm run build        # Build the Apptainer image from Docker Hub
pnpm run start        # Start the Qdrant server
```

### Directory Structure

```
packages/qdrant/
├── config/           # Qdrant configuration files
│   └── qdrant.yaml   # Main Qdrant configuration
├── storage/          # Persistent vector data storage
├── snapshots/        # Database snapshots
├── logs/             # Server logs
└── qdrant.sif        # Apptainer container image (generated)
```

### Data Persistence

All data is stored in the following directories (excluded from git):

- `storage/` - Vector collections and indices
- `snapshots/` - Database backups
- `logs/` - Runtime logs

These directories are automatically created and bound to the container at runtime.

## Apptainer Details

This package uses [Apptainer](https://apptainer.org/) (formerly Singularity) to run Qdrant in a containerized environment. Apptainer is particularly well-suited for HPC clusters and provides:

- Reproducible container environments
- No root daemon requirement
- Native integration with cluster schedulers
- Efficient storage with SIF format

The container image is built from the official [Qdrant Docker image](https://hub.docker.com/r/qdrant/qdrant) and configured for optimal performance in cluster environments.

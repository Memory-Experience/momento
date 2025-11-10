# AI Models

The API package leverages three categories of local AI models that run without external API dependencies: speech-to-text, text embeddings, and text generation. All models are designed for local inference with optional GPU acceleration.

## Design Philosophy

The model layer embodies several key architectural decisions:

- **Interface-driven design**: All models implement clear interfaces (`TranscriberInterface`, `EmbeddingModelInterface`, `LLMModelInterface`), enabling easy substitution and testing
- **Local-first approach**: No external API dependencies ensures privacy, offline operation, and predictable costs
- **Quantization strategy**: GGUF format models balance size and quality, making local inference practical
- **Dependency injection**: Models are injected into services rather than hard-coded, supporting multiple configurations

## Speech-to-Text

### Faster Whisper Transcriber

**Implementation**: `models/transcription/faster_whisper_transcriber.py`

Uses OpenAI's Whisper model optimized with CTranslate2 for real-time transcription. The default "small.en" model provides a good balance between accuracy and speed for local inference.

**Library**: [faster-whisper on GitHub](https://github.com/SYSTRAN/faster-whisper)

#### Key Design Decisions

**Streaming architecture**: Audio is processed in overlapping chunks (~2 seconds) with a small overlap (0.1s) maintained between chunks to ensure continuity across boundaries. This prevents word breaks at arbitrary chunk boundaries.

**Buffering strategy**: Rather than transcribing every audio packet immediately, the system accumulates enough audio to provide the model with sufficient context. This improves accuracy at the cost of slight latency.

**Format conversion**: Audio arrives as int16 PCM but Whisper expects float32 normalized to [-1, 1]. The conversion happens in the streaming pipeline to keep the interface clean.

### Alternative: Simul Streaming Transcriber

An experimental alternative using simultaneous translation techniques. Available but not currently used in production due to stability concerns.

## Embedding Models

### Qwen3 Embedding Model

**Implementation**: `models/embedding/qwen3_embedding.py`

Uses Alibaba's Qwen3-Embedding-0.6B model in quantized GGUF format for semantic embeddings.

**Model**: [Qwen3-Embedding-0.6B-GGUF on HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)

#### Key Design Decisions

**Quantization choice**: The Q8_0 (8-bit) quantization provides a good trade-off between model size and embedding quality. While more aggressive quantization (Q4) would reduce size further, embeddings are particularly sensitive to quantization artifacts since small changes in embedding space affect retrieval quality.

**llama.cpp backend**: Using llama.cpp for embeddings (not just generation) provides consistency across the model stack and leverages the same GPU acceleration infrastructure. This simplifies deployment and reduces dependencies.

**L2 normalization**: Embeddings are L2-normalized by default, which is standard practice for cosine similarity-based retrieval. This ensures all embeddings lie on a unit hypersphere, making distances directly comparable.

### Alternative Implementations

**SBert (all-MiniLM-L6-v2)**: A smaller embedding model available via HuggingFace Transformers. Used primarily for baseline comparisons and scenarios where lower resource usage is prioritized.

## Text Generation (LLM)

### Qwen3 Instruct Model

**Implementation**: `models/llm/qwen3.py`

Uses Alibaba's Qwen3-1.7B-Instruct model in quantized GGUF format for answer generation.

**Model**: [Qwen3-1.7B-GGUF on HuggingFace](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)

#### Key Design Decisions

**Streaming architecture**: The LLM generates tokens asynchronously and yields them in configurable chunks (default: 8 tokens). This design choice balances responsiveness with UI smoothness: smaller chunks feel more responsive but can appear choppy, while larger chunks provide smoother text rendering at the cost of perceived latency.

**Chat template integration**: The model expects a specific chat format with system/user/assistant roles. The `generate_with_memory()` method encapsulates this formatting, keeping prompt engineering concerns separate from the service layer.

**Context window management**: While the model supports 32K tokens, practical usage keeps contexts smaller (2-4K tokens) to balance quality and speed. Extremely long contexts can degrade generation quality and increase latency significantly.

**Grounding strategy**: The system prompt explicitly instructs the model to answer only from provided memories and acknowledge when it cannot answer. This reduces hallucination risk, though it's not foolproof (the threshold filtering service provides an additional safeguard).

### Alternative Implementations

**Generic llama.cpp wrapper**: A generic `LlamaCppModel` implementation supports any GGUF-format model, enabling experimentation with different LLMs (Llama 3, Mistral, Phi-3, etc.) without code changes.

## Model Loading Strategy

### Eager Initialization Pattern

All models are loaded at application startup rather than on first use. This design choice prioritizes request latency over startup time: the ~5-10 second cold start happens once, but every request benefits from immediate model availability.

```python
# At startup (container creation)
transcriber = FasterWhisperTranscriber()
transcriber.initialize()  # Loads model immediately
```

### GPU Acceleration

Models automatically detect and utilize available GPU acceleration (CUDA for NVIDIA, Metal for Apple Silicon). The `n_gpu_layers` parameter controls how many transformer layers are offloaded to the GPU:

- `-1` (default): Offload all layers to GPU
- `0`: CPU-only mode (no GPU offload)
- Positive number: Offload that specific number of layers

Higher values provide better performance at the cost of VRAM usage. This allows hybrid CPU/GPU inference when VRAM is limited: critical layers on GPU, remaining layers on CPU.

The decision to support flexible GPU offloading reflects the system's goal of being deployable in various environments: from personal laptops with limited VRAM to dedicated servers with high-end GPUs.

## Interface-Driven Architecture

All model categories are defined by interfaces (using Python's Protocol), not concrete implementations:

- **`TranscriberInterface`**: Defines `transcribe()` method signature
- **`EmbeddingModelInterface`**: Defines `embed()` and `embed_batch()` methods
- **`LLMModelInterface`**: Defines async `generate_with_memory()` method

**Why this matters**: Services depend on these interfaces, not specific implementations. This means:

1. **Testing**: Mock implementations can replace real models in tests without changing service code
2. **Flexibility**: Swapping Qwen3 for Llama 3 requires no changes to `LLMRAGService`
3. **Composition**: Multiple implementations can coexist (useful for evaluation and comparison)

This is a core application of the **Dependency Inversion Principle**: high-level services depend on abstractions, and concrete models are implementation details that can vary.

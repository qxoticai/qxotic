# [Quixotic AI](https://qxotic.ai)

Umbrella project for end-to-end LLM inference implementation for the JVM.
The goal is high performance, matching native C/C++/CUDA/Metal/HIP where it matters.

## Components

- [Jota](https://github.com/qxoticai/qxotic/tree/main/jota) Tensor engine
- [GGUF](https://github.com/qxoticai/qxotic/tree/main/gguf) llama.cpp's model format
- [Safetensors](https://github.com/qxoticai/qxotic/tree/main/safetensors) HuggingFace's model format
- [Tokenizers](https://github.com/qxoticai/qxotic/tree/main/tokenizers) supports TikToken and common tokenizers

All components needed for inference as standalone libraries:
- Model formats: **GGUF** and **Safetensors** both reading + writing
- Tokenization: **TikToken**, BPE and common tokenizers
- Tensor operations: Pristine [Tensor API](https://github.com/qxoticai/qxotic/blob/main/jota/jota-core/src/main/java/com/qxotic/jota/tensor/Tensor.java) with support for multiple-backends

## Java for AI?

Today "Java for AI" means:
- [LangChain4j](https://docs.langchain4j.dev)
- HTTP calls to external inference services
- JNI wrappers around native libraries

Many use cases can be satisfied locally, embeddings, small models... there's no reason for these to use Python or native code, the JVM provides superior performance and can offload work to
accelerators if needed.

## Status

**Experimental and early-stage.**

Can already run Llama, but there is still a lot to build and tune.

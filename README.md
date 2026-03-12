# Quixotic AI (qxotic.ai)

Umbrella project for end-to-end LLM inference implementation for the JVM.
The goal is high performance, matching native C/C++/CUDA/Metal/HIP where it matters.

## Components

- [GGUF](https://github.com/qxoticai/qxotic/tree/main/gguf)
- [Safetensors](https://github.com/qxoticai/qxotic/tree/main/safetensors)
- [Tokenizers](https://github.com/qxoticai/qxotic/tree/main/tokenizers)
- [Jota](https://github.com/qxoticai/qxotic/tree/main/jota)

Include all the core components needed for inference as standalone libraries:
- Model formats: **GGUF** and **Safetensors** (read + write)
- Tokenization: **TikToken** and common tokenizers, with high performance
- Tensor engine: **Tensor** API that runs on different devices
- 

## Java for AI?

Today "Java for AI" means:
- [LangChain4j](https://docs.langchain4j.dev)
- HTTP calls to external inference services
- JNI wrappers around native libraries

Many use cases that can be satisfied locally, there's no reason for these relegated to Python, the JVM provides superior performance and can also
run on other devices.

## Status

**Experimental and early-stage.**

Can already run Llama, but there is still a lot to build and tune.

---
sidebar_position: 1
---

# Quixotic AI

**AI sovereignty for the JVM.** A complete, open stack for running AI locally in Java:
tokenization, model formats, tensor math, native quantized matmul, and a full inference engine.

**No Python. No ONNX. No external services. Just jars.**

## The stack

| Library | What it does | Mantra |
|---------|--------------|--------|
| [jinfer](/jinfer) | In-process LLM inference: chat, vision, embeddings, reranking, speech, OpenAI-compatible server | **Local LLM inference for the JVM** |
| [jam](/jam) | Native quantized matmul, `R = W @ Aᵀ`, from Java or C | **Just a matmul.** The fastest one on the JVM |
| [jota](/jota) | Tensor engine with pluggable CPU/GPU backends | **One tensor API, every backend** |
| [toknroll](/toknroll) | LLM tokenization (tiktoken + SentencePiece BPE) | **Token-perfect** parity with the reference tokenizers |
| [gguf](/gguf) | Read/write GGUF model files | llama.cpp's format, pure Java |
| [safetensors](/safetensors) | Read/write Safetensors model files | HuggingFace's format, pure Java |
| [json](/json) | RFC 8259 JSON parser and printer | **~10 KB, zero dependencies, reflection-free** |

## Where to start

- **Building an application?** Pick a framework: [LangChain4j](/jinfer/langchain4j) or
  [Spring AI](/jinfer/spring-ai), or drop to the [jinfer CLI and server](/jinfer).
- **Need a piece, not the stack?** Every library stands alone: [toknroll](/toknroll) for
  tokenization, [gguf](/gguf) / [safetensors](/safetensors) for model files, [json](/json) for
  JSON, [jam](/jam) for matmul, [jota](/jota) for tensors.

## Conventions

- **Model strings.** Wherever a model is expected, a local path or hub reference
  (`user/repo:file`, or `modelscope.cn/user/repo:file` for another source) works. Downloads are resumable,
  checksum-verified, and cached, so warm runs never touch the network.
- **GGUF, everywhere.** Quantized weights, metadata and tokenizer in one file. `gguf` reads the
  layout, `toknroll` loads the tokenizer, `jinfer` runs the model.
- **Memory-first.** Tensors and activations live in `MemoryView` (jota), shared by jota, jam and
  jinfer, with no copies between layers.
- **Native-image first.** Every library compiles with GraalVM Native Image out of the box.

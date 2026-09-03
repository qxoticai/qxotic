<h1 align="center"><a href="https://qxotic.ai">Quixotic AI</a></h1>

<p align="center"><strong>AI sovereignty for the JVM</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
  <a href="https://x.com/qxoticai"><img src="https://img.shields.io/badge/-grey?logo=X" alt="Qxotic AI on X"></a>
  <a href="https://bsky.app/profile/qxotic.ai"><img src="https://img.shields.io/badge/-grey?logo=bluesky&logoColor=f5f5f5" alt="Qxotic AI on Bluesky"></a>
</p>

Quixotic AI provides a complete, open stack, for local AI on the JVM Java. From tokenizers and model formats, to a full inference engine with multi-modal capabilities.

**No Python. No ONNX. No external services. Just AI, in a jar.**

--- 

## Features

- **Designed from first-principles for the JVM.** AI runs end-to-end on the JVM. No sidecar servers, no ONNX, no IPC, no Python.
- **Write once, accelerate everywhere.** A common tensor API for CPUs and GPUs.
- **Optional native acceleration.** Fast matrix multiplication routines, competitive with `llama.cpp`.
- **GraalVM's Native Image.** First-class support for GraalVM Native Image: small footprint, millisecond startup, self-contained binaries.

---

## The Quixotic AI stack

| Module | What it is | One-liner |
|--------|------------|-----------|
| [`jinfer`](./jinfer) | Inference engine | **Local AI inference for the JVM.** Chat, vision, audio, embeddings, reranking, text-to-speech 
| [`toknroll`](./toknroll) | LLM tokenization | **Token-perfect.** Fast tokenizers for LLMs, pure Java, zero dependencies |
| [`jam`](./jam) | Quantized matrix multiplication | **Just a matmul.** Native implementations for several CPU ISAs |
| [`jota`](./jota) | Tensor engine | **Write once, accelerate everywhere.** Java, C, CUDA, HIP, Metal, OpenCL, Mojo |
| [`gguf`](./gguf) | GGUF reader/writer | llama.cpp's model format, pure Java, zero dependencies |
| [`safetensors`](./safetensors) | Safetensors reader/writer | HuggingFace's model format, pure Java, zero dependencies |

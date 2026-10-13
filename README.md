<h1 align="center"><a href="https://qxotic.ai">Quixotic AI</a></h1>

<p align="center"><strong>AI sovereignty for the JVM</strong></p>

<p align="center">
  <a href="https://central.sonatype.com/namespace/com.qxotic"><img src="https://img.shields.io/maven-central/v/com.qxotic/jinfer-bom?label=Maven%20Central" alt="Maven Central"></a>
  <a href="https://www.youtube.com/watch?v=S7x4DxwRnk4"><img src="https://img.shields.io/badge/Devoxx_Belgium_2026-talk-FF0000?logo=youtube&logoColor=white" alt="Devoxx Belgium 2026 talk"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
  <a href="https://x.com/qxoticai"><img src="https://img.shields.io/badge/-grey?logo=X" alt="Qxotic AI on X"></a>
  <a href="https://bsky.app/profile/qxotic.ai"><img src="https://img.shields.io/badge/-grey?logo=bluesky&logoColor=f5f5f5" alt="Qxotic AI on Bluesky"></a>
</p>

<div align="center">

Quixotic AI provides a complete, open stack for local AI on the JVM.<br>
From tokenizers and model formats to a full inference engine with multi-modal capabilities.

_No Python. No ONNX. No external services. Just_ **AI, in a jar.**

</div>

## Highlights

- **Designed from first-principles for the JVM.** AI runs end-to-end on the JVM. No sidecar servers, no ONNX, no IPC, no Python.
- **Write once, accelerate everywhere.** A common tensor API for CPUs and GPUs.
- **Optional native acceleration.** Fast matrix multiplication routines, competitive with `llama.cpp`.
- **GraalVM's Native Image.** First-class support for GraalVM Native Image: self-contained binaries, with small footprint and millisecond startup.

## [Quixotic: Pure Java Inference at Native Speed](https://www.youtube.com/watch?v=S7x4DxwRnk4)

[![Watch "Quixotic: Pure Java Inference at Native Speed" at Devoxx Belgium 2026](https://img.youtube.com/vi/S7x4DxwRnk4/maxresdefault.jpg)](https://www.youtube.com/watch?v=S7x4DxwRnk4)

<p align="center"><strong>Presented at Devoxx Belgium 2026</strong></p>

## The Quixotic AI stack

| Module | What it is | One-liner |
|--------|------------|-----------|
| [`jinfer`](./jinfer) | AI inference engine | **Local AI inference for the JVM.** Chat, vision, audio, embeddings, reranking, text-to-speech |
| [`toknroll`](./toknroll) | LLM tokenization | **Token-perfect.** Fast tokenizers for LLMs, pure Java, zero dependencies |
| [`jam`](./jam) | Quantized matrix multiplication | **Just a matmul.** Native implementations for several CPU ISAs |
| [`jota`](./jota) | Tensor engine | **Write once, accelerate everywhere.** Java, C, CUDA, HIP, Metal, OpenCL, Mojo |
| [`gguf`](./gguf) | GGUF reader/writer | llama.cpp's model format, pure Java, zero dependencies |
| [`safetensors`](./safetensors) | Safetensors reader/writer | HuggingFace's model format, pure Java, zero dependencies |
| [`json`](./json) | JSON parser and printer | RFC 8259, ~25 KB, zero dependencies, zero reflection |

## Demos

The demos need a JDK 25 and [JBang](https://www.jbang.dev/). The first run downloads the CLI and the model.

### Chat

```bash
jbang jinfer@qxoticai chat --model LiquidAI/LFM2.5-350M-GGUF:Q8_0
```

### OpenAI-compatible server

```bash
jbang jinfer@qxoticai --model openbmb/MiniCPM5-2B-GGUF:Q4_K_M --context-capacity 16384 server
```

Then, from another terminal, any OpenAI client works against `http://localhost:54154/v1`:

```bash
curl -s http://localhost:54154/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

### Live transcription

<p align="center">
  <a href="jinfer/jinfer-parakeet"><img src="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.gif" alt="Live speech transcription with jinfer-parakeet" width="720"></a>
</p>
<p align="center"><sub>Live transcription with <a href="jinfer/jinfer-parakeet">jinfer-parakeet</a>, running on the JVM.</sub></p>

**Try it yourself.** Run the command for your platform, start talking, and watch the words appear as you speak.
[ffmpeg](https://ffmpeg.org/) captures the microphone:

```bash
# Linux
ffmpeg -nostats -loglevel error -f pulse -i default -ar 16000 -ac 1 -f s16le - \
  | jbang jinfer@qxoticai -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf transcribe - --raw-pcm

# macOS
ffmpeg -nostats -loglevel error -f avfoundation -i ":default" -ar 16000 -ac 1 -f s16le - \
  | jbang jinfer@qxoticai -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf transcribe - --raw-pcm
```

On Linux and macOS these use the system's default microphone.
On Windows, ffmpeg has no default input, so it needs the microphone's name. List the devices first:

```bat
ffmpeg -hide_banner -list_devices true -f dshow -i dummy
```

Then put the name of the `(audio)` entry in `audio="..."`, and run from `cmd` or PowerShell 7.4+:

```bat
ffmpeg -nostats -loglevel error -f dshow -i audio="Microphone (Realtek(R) Audio)" -ar 16000 -ac 1 -f s16le - ^
  | jbang jinfer@qxoticai -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf transcribe - --raw-pcm
```

## Build and test

Requires a JDK 25 and Maven 3.9; cmake and a C compiler to build the native jam kernels.

```bash
make test-fixtures   # once after cloning: tokenizer vocabularies, golden truth, enwik8, Parakeet speech
make test            # the default suite: no models, no network
make ci              # what a pull request runs: formatting, the suite, the corpus tests, the release shape
```

`make help` lists the rest.
The suites that need models or hardware are opt-in; each module's README describes how to run them.  
See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

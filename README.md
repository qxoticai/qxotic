<h1 align="center"><a href="https://qxotic.ai">Quixotic AI</a></h1>

<p align="center"><strong>AI sovereignty for the JVM</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
  <a href="https://x.com/qxoticai"><img src="https://img.shields.io/badge/-grey?logo=X" alt="Qxotic AI on X"></a>
  <a href="https://bsky.app/profile/qxotic.ai"><img src="https://img.shields.io/badge/-grey?logo=bluesky&logoColor=f5f5f5" alt="Qxotic AI on Bluesky"></a>
</p>

Quixotic AI is a complete, open stack for local AI in Java, from tokenizers and model formats up to a full inference engine with chat, vision,
embeddings, reranking and speech.

**No Python. No ONNX. No external services. Just AI, in a jar.**

---

## Why Quixotic

- **In-process, by design.** Models load, tokenize and generate inside the JVM. No sidecar servers,
  no IPC, no Python runtime to keep alive.
- **The whole path is open.** Every layer is here and readable in Java: tokenizer, model format,
  tensor math, kernels, engine. Nothing in the stack is an opaque binary.
- **Write once, accelerate everywhere.** A single tensor API across CPU and GPU backends. Switch
  backends with one line, or none at all, since discovery is automatic.
- **Fast where it counts.** Hand-tuned SIMD kernels (SSE3 → AVX-512-VNNI, NEON → I8MM, Apple Metal)
  that match, and often beat, llama.cpp at equal ISA.
- **Native-image first.** Every module is GraalVM-ready out of the box: small footprint,
  millisecond startup, single self-contained binary.
- **Apache 2.0.** No API key, no quota, no service to call.

---

## Quick start

Talk to a model in one file, with no build required. [JBang](https://www.jbang.dev/) resolves
everything:

```bash
cd jinfer/examples/scripts
jbang Chat.java "Invent a tiny language for talking to houseplants."   # streaming chat
jbang Json.java "Ada Lovelace, born 1815 in London."                   # schema-perfect JSON
jbang Narrate.java photo.jpg                                           # vision and speech
```

Models download once from Hugging Face and stay in the local cache. To use the engine from an
application, add two dependencies and call a builder. The [jinfer README](./jinfer/README.md) has
the details.

---

## The stack

Each module stands alone and has its own README. Start at `jinfer` to run a model; the layers below
it are usable on their own.

| Module | What it is | One-liner |
|--------|------------|-----------|
| [`jinfer`](./jinfer) | Inference engine | **Local AI inference for the JVM.** Chat, vision, embeddings, reranking, speech |
| [`jam`](./jam) | Quantized matrix multiplication | **Just a matmul.** The fastest one on the JVM |
| [`jota`](./jota) | Tensor engine | **One tensor API, every backend.** Panama, C, CUDA, HIP, Metal, OpenCL, Mojo |
| [`toknroll`](./toknroll) | LLM tokenization | **Token-perfect.** Byte-exact parity with the reference tokenizers |
| [`gguf`](./gguf) | GGUF reader/writer | llama.cpp's model format, pure Java, zero dependencies |
| [`safetensors`](./safetensors) | Safetensors reader/writer | HuggingFace's model format, pure Java, zero dependencies |
| [`json`](./json) | JSON parser/printer | **~10 KB, zero dependencies, reflection-free.** Jackson not required |

---

## Where to go next

| To do this | Read |
|------------|------|
| Run a model, from Java or the CLI | [jinfer](./jinfer/README.md) |
| See which model families and quantizations run | [jinfer, What runs](./jinfer/README.md#what-runs) |
| Write LangChain4j code | [jinfer-langchain4j](./jinfer/jinfer-langchain4j/README.md) |
| Write Spring AI or Spring Boot code | [jinfer-spring-ai](./jinfer/jinfer-spring-ai/README.md) |
| Serve an OpenAI-compatible endpoint | [jinfer, CLI and server](./jinfer/README.md#cli-and-server) |
| Use the tensor API or add a backend | [jota](./jota/README.md) |
| Read or write model files directly | [gguf](./gguf/README.md), [safetensors](./safetensors/README.md) |

Full documentation is at [qxotic.ai](https://qxotic.ai).

---

## Building

Java 25, one Maven reactor, one entry point:

```bash
make help        # every build entry point
make jar         # the jinfer CLI jar
make native      # a GraalVM native binary (jinfer/jinfer)
make test        # the full suite
```

The Makefiles wrap the Maven reactor. Use it directly for finer selection:

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests   # the CLI and everything it needs
mvn package                                          # everything
```

`jota` is dependency-closed (`mvn -f jota/pom.xml` works standalone); the other subtrees build from
the root. Details live in each module's README.

---

## License

Apache 2.0

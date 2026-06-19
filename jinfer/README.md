# LFM25.java

<p align="center">
  <img src="https://github.com/user-attachments/assets/54f57e18-b26e-4121-8ef8-9522f28ad0b4">
</p>

<div align="center">

![Java 21+](https://img.shields.io/badge/Java-21%2B-007396?logo=java&logoColor=white)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

Fast, zero-dependency, inference engine for [Liquid AI](https://www.liquid.ai/) [LFM2.5 models](https://www.liquid.ai/models) in pure Java.

</div>

----

## Features

- Based on [llama3.java](https://github.com/mukel/llama3.java)
- Multi-architecture behind a small `Model` interface (each model is one implementation):
  - Liquid AI **LFM2.5** GGUF models (dense and MoE, with short-convolution layers)
  - Google **Gemma 4** GGUF models — the per-layer-embedding E-series (E2B/E4B) and the
    mixture-of-experts A4B, with sliding-window attention and logit soft-capping
  - Meta **Llama 3.x** GGUF models (the standard Llama transformer, with "llama3" RoPE
    frequency scaling) — also covers **MiniCPM** (same architecture plus three scalars),
    **Mistral-3 / Ministral** (plus YaRN RoPE + attention-temperature scaling), and
    **IBM Granite 4.1** (dense; plus four scalars incl. a custom QK attention scale)
  - **Qwen 3.5** (dense and MoE), OpenAI **gpt-oss** (MXFP4 experts), and NVIDIA **Nemotron**
    (hybrid Mamba2 + attention + MoE)
- Fast [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser
- Supported dtypes/quantizations: `F16`, `BF16`, `F32`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`
- Fast kernels using Java's [Vector API](https://openjdk.org/jeps/469)
- CLI with `--chat`, `--prompt`, and OpenAI-compatible `--server` modes
- Thinking mode control with `--think off|on|inline`
- GraalVM Native Image support
- AOT model preloading for **instant time-to-first-token**

## Setup

Download an [LFM2.5 model](https://www.liquid.ai/models) in GGUF format or convert one with [llama.cpp](https://github.com/ggml-org/llama.cpp). The runner expects a `.gguf` file compatible with LFM2.5 metadata.

#### Optional: pure quantizations

`Q4_0` files are often mixed-quant in practice. A pure quantization is not required, but can be generated from an F32/F16/BF16 GGUF source with `llama-quantize` from [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
./llama-quantize --pure ./LFM2.5-1.2B-Instruct-BF16.gguf ./LFM2.5-1.2B-Instruct-Q4_0.gguf Q4_0
```

Pick any supported target quantization, for example `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, or `Q8_0`.

## Build and run

Java 25 is required (the build targets release 25 with `--enable-preview`), in particular for the
[`MemorySegment` mmap feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).

Build the fat jar with Maven (or `make jar`):

```bash
mvn package            # -> target/lfm25.jar
```

Run it (the incubator/native-access flags are required at run time):

```bash
JAVA_FLAGS="--enable-preview --add-modules jdk.incubator.vector,jdk.httpserver \
  --enable-native-access=ALL-UNNAMED -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0"

java $JAVA_FLAGS -jar target/lfm25.jar --help
java $JAVA_FLAGS -jar target/lfm25.jar --model ./LFM2.5-1.2B-Instruct-Q8_0.gguf --chat
java $JAVA_FLAGS -jar target/lfm25.jar --model ./LFM2.5-1.2B-Instruct-Q8_0.gguf --prompt "Tell me a joke"
java $JAVA_FLAGS -jar target/lfm25.jar --model ./LFM2.5-1.2B-Instruct-Q8_0.gguf --server --port 17325
```

`mvn -Pnative package` (GraalVM ≥ 25.0.3) or `make native` produces a self-contained `lfm25` binary
with no runtime flags required.

## CLI

```text
Usage:  java -jar lfm25.jar [options]

Options:
  --model, -m <path>            required, path to .gguf file
  --interactive, --chat, -i     run in chat mode
  --instruct                    run in instruct (once) mode, default mode
  --server                      run an OpenAI-compatible HTTP server
  --host <host>                 server bind host, default 127.0.0.1
  --port <int>                  server bind port, default 17325
  --prompt, -p <string>         input prompt
  --suffix <string>             suffix for fill-in-the-middle request
  --system-prompt, -sp <string> system prompt for chat/instruct mode
  --temperature, -temp <float>  temperature in [0,inf], default 1.0
  --top-p <float>               p value in top-p sampling in [0,1], default 0.95
  --seed <long>                 random seed, default System.nanoTime()
  --max-tokens, -n <int>        number of steps to run, default 1024
  --stream <boolean>            print tokens during generation, default true
  --echo <boolean>              print all tokens to stderr, default false
  --color <on|off|auto>         colorize thinking output in terminal, default auto
  --think <off|on|inline>       control thinking output
  --keep-past-thinking <bool>   keep prior assistant thinking in history, default false
  --raw-prompt                  bypass chat template and tokenize --prompt directly
```

### OpenAI-compatible server

Start the server:

```bash
java $JAVA_FLAGS -jar target/lfm25.jar --model ./LFM2.5-1.2B-Instruct-Q8_0.gguf --server --host 127.0.0.1 --port 17325
```

Then call `/v1/chat/completions` with any OpenAI-compatible client:

```bash
curl http://127.0.0.1:17325/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm25","messages":[{"role":"user","content":"Tell me a joke"}],"max_tokens":128}'
```

The server exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Both completion endpoints support `stream: true`, `temperature`, `top_p`, `seed`, `max_tokens`, `max_completion_tokens`, and `stop`. Only `n: 1` is supported.

Chat completions support OpenAI-style function `tools` and `tool_choice` values `auto`, `none`, `required`, or a named function choice object. Tool calling is prompt-guided: when a tool is selected, the model is instructed to emit JSON like:

```json
{"tool_calls":[{"name":"get_weather","arguments":{"location":"Paris"}}]}
```

The server converts that JSON to OpenAI's `message.tool_calls` response format with `finish_reason: "tool_calls"`. Follow-up requests can include assistant `tool_calls` messages and `role: "tool"` results. Legacy assistant `function_call` messages and model outputs are also accepted and normalized to `tool_calls` responses.

### GraalVM Native Image

Compile with `make native` to produce a `lfm25` executable, then:

```bash
./lfm25 --model ./LFM2.5-8B-A1B-Q8_0.gguf --chat
```

### AOT model preloading

`LFM25.java` supports AOT model preloading to reduce parse overhead and time-to-first-token (TTFT).

To AOT pre-load a GGUF model:
```bash
PRELOAD_GGUF=/path/to/model.gguf make native
```

A larger specialized binary is generated with parse overhead removed for that specific model.
It can still run other models with the usual parsing overhead.

## Benchmarks

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c22e2a-2003-424c-aa27-e27737880c33">
</p>

\*\**Hardware specs: AMD Ryzen 9950X 16C/32T 64GB (6400) Linux 6.18.12.*

[GraalVM 25+](https://www.graalvm.org/downloads) is recommended for the absolute best performance (JIT mode), it provides partial, but good support for the [Vector API](https://openjdk.org/jeps/469), also in Native Image.

By default, the "preferred" vector size is used, it can be force-set with `-Dllama.VectorBitSize=0|128|256|512`, `0` means disabled.

## Related Repositories

- [llama3.java](https://github.com/mukel/llama3.java)
- [gemma4.java](https://github.com/mukel/gemma4.java)
- [gptoss.java](https://github.com/mukel/gptoss.java)
- [qwen35.java](https://github.com/mukel/qwen35.java)
- [nemotron3.java](https://github.com/mukel/nemotron3.java)

## License

Apache 2.0

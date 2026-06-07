# LFM25.java

<div align="center">

![Java 21+](https://img.shields.io/badge/Java-21%2B-007396?logo=java&logoColor=white)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

Fast, small, inference engine for [Liquid AI](https://www.liquid.ai/) [LFM2.5 models](https://www.liquid.ai/models) in pure Java.

</div>

----

## Features

- Single file GGUF runner, based on [llama3.java](https://github.com/mukel/llama3.java)
- Supports Liquid AI LFM2.5 GGUF models
- Fast [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser
- Supported dtypes/quantizations: `F16`, `BF16`, `F32`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`
- Matrix-vector kernels using Java's [Vector API](https://openjdk.org/jeps/469)
- LFM2.5 architecture support: attention, short convolution layers, SWA, shared KV, dense FFN, and MoE FFN
- CLI with `--chat`, `--prompt`, and raw prompt modes
- Thinking mode control with `--think off|on|inline`

## Setup

Download an [LFM2.5 model](https://www.liquid.ai/models) in GGUF format or convert one with [llama.cpp](https://github.com/ggml-org/llama.cpp). The runner expects a `.gguf` file compatible with LFM2.5 metadata.

#### Optional: pure quantizations

`Q4_0` files are often mixed-quant in practice. A pure quantization is not required, but can be generated from an F32/F16/BF16 GGUF source with `llama-quantize` from [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
./llama-quantize --pure ./LFM2.5-1.2B-Instruct-BF16.gguf ./LFM2.5-1.2B-Instruct-Q4_0.gguf Q4_0
```

Pick any supported target quantization, for example `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, or `Q8_0`.

## Build and run

Java 21+ is required, in particular for the [`MemorySegment` mmap feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).

[`jbang`](https://www.jbang.dev/) is a good fit for this use case.

```bash
jbang LFM25.java --help
jbang LFM25.java --model ./LFM2.5-1.2B-Instruct-Q8_0.gguf --chat
jbang LFM25.java --model ./LFM2.5-1.2B-Instruct-Q8_0.gguf --prompt "Tell me a joke"
```

Or run it directly, still via [`jbang`](https://www.jbang.dev/):

```bash
chmod +x LFM25.java
./LFM25.java --help
```

## CLI

```text
Usage:  jbang LFM25.java [options]

Options:
  --model, -m <path>            required, path to .gguf file
  --interactive, --chat, -i     run in chat mode
  --instruct                    run in instruct (once) mode, default mode
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

## Performance

[GraalVM 25+](https://www.graalvm.org/downloads) is recommended for the best performance. It provides good support for the [Vector API](https://openjdk.org/jeps/469), including Native Image support.

By default, the preferred vector size is used. It can be force-set with `-Dllama.VectorBitSize=0|128|256|512`, where `0` disables Vector API kernels.

## Related Repositories

- [llama3.java](https://github.com/mukel/llama3.java)
- [gemma4.java](https://github.com/mukel/gemma4.java)
- [gptoss.java](https://github.com/mukel/gptoss.java)
- [qwen35.java](https://github.com/mukel/qwen35.java)
- [nemotron3.java](https://github.com/mukel/nemotron3.java)

## License

Apache 2.0

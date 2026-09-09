# Jinfer Bench

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

The benchmark harness for the [Jinfer](../README.md) inference engine.
This page documents the workloads, the commands, and the rules for every throughput number Jinfer reports.
`JinferBench` runs the same workload as `llama-bench`.
Numbers from both are comparable when the model, quantization, thread count, and machine are the same.

## Run it

The harness runs as a native image or on the JVM.
The native image measures what the shipped binary does.
The JVM run measures what the jars do.
The first line of the output names which one ran.

Native image, from the repository root (requires GraalVM Native Image 25.0.3 or newer; the image is built for the current machine):

```bash
make -C jinfer/jinfer-bench native            # -> bin/jinfer-bench
bin/jinfer-bench -m model.gguf -p 512 -n 128 -r 5 -w 2
```

JVM, from the repository root (JDK 25):

```bash
mvn -pl jinfer/jinfer-bench -am -DskipTests package
java --add-modules jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
  -jar jinfer/jinfer-bench/target/jinfer-bench.jar -m model.gguf -p 512 -n 128 -r 5 -w 2
```

`-m` takes a path to a GGUF file, not a model reference.
`jinfer pull owner/repo:QUANT` downloads a model and prints its path.
The two JVM flags are the ones the native image is built with: the Vector API module, and no bounds checks inside vector loads.
Use no other flags for a reported number.

The output is one header line and two tables.
The following is a real run on a 16-core Zen 5 desktop, on the JVM and then on the native image:

```text
jinfer-bench 0.2.0 | runtime: GraalVM JIT | jam: native, vector, scalar | threads: 16 (engine default)
| model            | threads | test   |              t/s |
| ---------------- | ------: | ------ | ---------------: |
| LFM2.5-350M-Q8_0 |      16 | pp512  | 8551.38 ± 647.87 |
| LFM2.5-350M-Q8_0 |      16 | tg128  |    245.82 ± 1.72 |

| model            | state ms | ttft ms | cache hit ms                        | mtp           | media cold/warm ms | peak MB |
| ---------------- | -------- | ------- | ----------------------------------- | ------------- | ------------------ | ------- |
| LFM2.5-350M-Q8_0 | 5.9      | 77.2    | 188.7 vs 475.6 (2.5x, 639 restored) | no draft head | n/a                | 1509    |
```

```text
jinfer-bench 0.2.0 | runtime: native image | jam: native, vector, scalar | threads: 16 (engine default)
| model            | threads | test   |              t/s |
| ---------------- | ------: | ------ | ---------------: |
| LFM2.5-350M-Q8_0 |      16 | pp512  | 6104.32 ± 416.37 |
| LFM2.5-350M-Q8_0 |      16 | tg128  |    241.85 ± 3.17 |
```

The header lists the version, the compiler, the jam backends that loaded (in the order matmuls try them), and the thread budget.
`t/s` is the mean and standard deviation of tokens per second over the timed repetitions.
`n/a` means the model does not have the capability, or the measurement was not requested.
Warm-up progress goes to stderr and the tables go to stdout, so `> results.md` captures only the tables.

## What the numbers are

The first table is the `llama-bench` workload, one row per test.

| test | what is timed |
|------|---------------|
| `pp512` | 512 synthetic tokens ingested in chunks of 512 (`llama-bench -ub 512`); one logits projection for the last token; context sized to the test |
| `tg128` | 128 single-token decodes from an empty state at positions 0 to 127; logits are projected every step and not argmaxed, matching `llama-bench`, which feeds back `rand() % n_vocab` |

Each test uses one state, reset before every warm-up pass and every timed repetition.
No allocation or page fault occurs inside a timed region.
Warm-up repeats the whole test until throughput settles within 3% over a window of three passes, with at least `-w` passes and at most 30.
A JIT needs several passes; native code needs one.
`-d N` keeps `N` prefix tokens resident, prepared outside the timing, like `llama-bench -d`.

The second table measures the engine layer through `ChatEngine` on a real prompt: state allocation, time to first token from a cold state, a prompt-cache hit compared with a full prefill, projected-media latency cold and warm (`--media`), and peak RSS.

**A draft head is never part of a throughput number.**
`pp512` and `tg128` drive the model at the state and batch level and do not use speculation.
The `mtp` cell is a separate measurement of draft acceptance and net decode speed-up.
It runs only when a draft head is attached with `--with speculation=<mtp.gguf>`; otherwise it reads `no draft head`.

## What decides the numbers

**Compiler.**
The native image and the GraalVM JIT compile the Vector API kernels fully.
C2 runs the quantized kernels mostly through the un-intrinsified per-lane fallback.
The engine reports this at startup as the `SLOW_JIT` cliff.
On a GraalVM JDK the Graal JIT is the default; `-XX:-UseJVMCICompiler` selects C2.
The header line names the compiler.
Do not compare a C2 number with a Graal number.

**Backend.**
Matmuls try the jam backends in this order: `native` (C kernels in `libjam`), `vector` (Java Vector API kernels), `scalar` (autovectorized Java).
If no backend applies, the engine's built-in Java kernels run.
Prefill uses the first backend that has a kernel for the weight type.
Decode uses the native gemv only where it is faster: on x86, at low thread counts (the crossover is between 4 and 16 threads, depending on the quantization).
`-Djinfer.q4.nativeDecode=true|false` (also `q8`, `mxfp4`, `kq`) forces the choice.
`-Djam.native.disabled=true` (or `vector`, `scalar`) removes a backend before it is probed.
This is how the Java kernels are compared against the native ones.
The header line lists the backends that remain.

**Threads.**
The default is one thread per physical core; on Apple Silicon, P-cores only, as in `llama-bench`.
`-t N` sets the same count for both tests and every backend.
Decode is memory-bound at high thread counts; threads beyond the crossover do not increase decode throughput and can reduce it.

**Warnings.**
The engine reports each performance cliff once per run, on stderr, as `perf cliff [NAME]: ...`.
A reported number must come from a run without cliffs, or state which cliff it carried.

| cliff | it means | do this |
|-------|----------|---------|
| `SLOW_JIT` | the JIT is C2; the Vector API kernels are compiled conservatively | run on the GraalVM JIT or the native image |
| `JAM_ABSENT` | no jam backend on the classpath; prefill runs on the Java kernels | use the shaded jar or the image; a custom classpath needs `jam-native` or `jam-vector` |
| `JAM_DECLINE` | the native kernels declined a matmul shape | the `libjam` build does not match; rebuild the natives and report the cliff if it persists |
| `NATIVE_ACCESS_RESTRICTED` | native access is restricted; vector accesses keep extra checks | add `--enable-native-access=ALL-UNNAMED`; the shaded jar's manifest sets it |
| `MAMBA2_SCALAR`, `GDN_SCALAR` | the model's recurrent scan has no vector kernel in this build | not a setup problem; report the cliff together with the number |

## Rules for a reported number

- Run on an idle machine on mains power, with no other workload.
  Decode is memory-bound; background load shows in `tg128`.
- Use one process per configuration.
  `convTile`, `vectorJit`, and the thread budget are read once at class initialization; a second configuration in the same process measures the first one's settings.
- Use the same thread count on both sides of a comparison, and state it.
- When comparing two engines or two configurations, run them in the order A, B, B, A so that thermal drift and background load affect both sides equally.
- Use at least `-r 5`.
  The standard deviation is part of the number.
- Do not report a single cold run; the first passes on a JVM measure the JIT, not the engine.
- Do not report a number from another machine; measure on the machine you describe.

The llama.cpp side of a comparison uses the same lengths, repetitions, and threads, with llama.cpp built for the same machine:

```bash
llama-bench -m model.gguf -p 512 -n 128 -r 5 -t 16
```

`llama-bench -b 2048` only caps the tokens per decode call; ggml computes 512-row micro-batches, the same as `pp512` here.

A report contains the machine, the header line, and the tables, verbatim:

```text
<CPU> (<cores> cores), <RAM>, <OS>, <JDK>, jinfer commit <sha>, idle
jinfer-bench 0.2.0 | runtime: native image | jam: native, vector, scalar | threads: 16 (engine default)
<the two tables>
llama-bench <build>, -t 16: pp512 <t/s>, tg128 <t/s>
```

## Reference

### JinferBench

| option | meaning |
|--------|---------|
| `-m, --model <path>` | GGUF file to benchmark, repeatable; any architecture on the classpath |
| `-p, --n-prompt <N>` | prefill tokens, default 512; `0` skips `pp` |
| `-n, --n-gen <N>` | decode tokens, default 128; `0` skips `tg` |
| `-d, --n-depth <N>` | resident prefix tokens, prepared outside the timing, default 0 |
| `-r, --repetitions <N>` | timed repetitions, default 5 |
| `-w, --warmup <N>` | minimum warm-up passes, default 2; warm-up continues until throughput settles |
| `--no-warmup` | skip warm-up; for native code only |
| `--no-capabilities` | skip the second table |
| `-t, --threads <N>` | thread budget for both tests and every backend; default: one per physical core |
| `--ctx <N>` | context for both tests; default: `p` for `pp`, `n` for `tg`, as `llama-bench` |
| `--with <capability>=<path>` | attach a companion, repeatable: `media=<mmproj.gguf>`, `speculation=<mtp.gguf>` |
| `--media <image>` | also measure projected-media cold and warm latency; needs `--with media=` |

Vision: attach the projector and measure an image:

```bash
bin/jinfer-bench -m gemma-4-12b-it-Q8_0.gguf -p 512 -n 128 -r 5 -w 2 \
  --with media=mmproj-F32.gguf --media cat.png
```

### EmbedBench

`EmbedBench` measures the packed-embedding path (`EmbeddingModel.embedAll`): variable-length sequences packed into segmented forward passes over one KV context, with each pooled vector streamed out.
The workload is deterministic: lengths from a multiplicative hash, greedy filler tokens.

```bash
java --add-modules jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
  -cp jinfer/jinfer-bench/target/jinfer-bench.jar com.qxotic.jinfer.bench.EmbedBench \
  -m embedder.gguf -s 256 --minlen 8 --maxlen 64 -b 512 -r 5 -w 3
```

| option | meaning |
|--------|---------|
| `-m, --model <path>` | embedding checkpoint; the port declares its pooling |
| `-s, --sequences <N>` | packed sequences, default 256 |
| `--minlen, --maxlen <N>` | ragged length range, default 8 to 64 |
| `-b, --batch <N>` | per-chunk forward width, default 512 |
| `-r, --repetitions <N>` | timed repetitions, default 5 |
| `-w, --warmup <N>` | minimum warm-up passes, default 3; then adaptive |
| `-t, --threads <N>` | thread budget |

It reports `tok/s` over the packed tokens and `seq/s`.
`llama-bench --embeddings 1` uses one flat 512-token prompt, so only `tok/s` is comparable with it.

### Microbenchmarks

Loops over engine kernels without a model; run them with the same `-cp` as `EmbedBench`.

- `SpinProbe [iters]`: the cost of one empty `Parallel` region, which is the dispatch and barrier latency of every parallel region in a decode token.
- `ConvPeak [census.log]`: `Convolutions.conv1dRows` throughput across the register-tile shapes.
  With a shape census from `-Djinfer.convProfile=true` on a real synthesis, it measures the shapes that model ran, weighted by FLOPs.
  The tile shape is a constant, so one process measures one shape; run it once per `-Djinfer.convTile=auto|4x2|4x4`.
- `ConvParity`: checks whether the tile shape changes the output; run it once per `-Djinfer.convTile` value and `diff` the outputs.

Model-level probes are tagged JUnit benches in the model modules: `MtpBench` (jinfer-gemma4); `PrefillBench`, `ScoringBench`, `GrammarCostProbe`, `NmtProbe` (jinfer-langchain4j).
Run them by name: `mvn -pl jinfer/jinfer-gemma4 test -Dtest=MtpBench -Dsurefire.excludedGroups=`.

### Measurement instruments

These change what is compiled or recorded.
They are for locating a regression, not for a reported number.
Each has its own documentation.

- [`hotspot_compile_commands`](../hotspot_compile_commands): C2 inlining hints for the hot Vector API helpers, to separate an inlining regression from a kernel regression under C2. The Graal JIT does not read it.
- `-Djinfer.convTile=auto|4x2|4x4` and `-Djinfer.convProfile=true`: the convolution register tile and its shape census; documented in `jinfer/pom.xml` and `Convolutions.TILE_CODE`.
- `-Djinfer.vectorJit=auto|fast|slow`: overrides the compiler detection behind `SLOW_JIT`.
- JFR (`-XX:StartFlightRecording`): start it after warm-up, or filter the samples by time; a recording that starts with the JVM attributes interpreter and C1 time to the kernels.

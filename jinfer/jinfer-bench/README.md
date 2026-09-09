# Jinfer Bench

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

The benchmark harness for the [Jinfer](../README.md) inference engine: the workloads, the commands and the rules behind every throughput number Jinfer reports.
`JinferBench` mirrors `llama-bench`, so a Jinfer number and a llama.cpp number for the same model, quant, thread count and machine are comparable.

## Run it

Two ways to run the same harness.
The native image is what a user of the shipped binary gets; the JVM run is what a user of the jars gets.
Report which one you ran - the first line of the output says so.

Native image, from the repository root (GraalVM Native Image 25.0.3 or newer, built for this machine):

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

`-m` takes a path to a GGUF file, not a model ref; `jinfer pull owner/repo:QUANT` prints the path of a downloaded one.
The JVM flags are the two the native image is built with: the Vector API module, and bounds checks off inside vector loads, which the image also folds away.
Nothing else goes on the command line for a reported number.

The output is one header line and two tables.
This is a real run on a 16-core Zen 5 desktop, first on the JVM, then the native image:

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

The header names what decided the numbers: the version, the compiler, the jam backends that loaded in the order the matmuls try them, and the thread budget.
`t/s` is the mean and standard deviation of tokens per second over the timed repetitions.
An `n/a` cell is a capability the model does not have or a measurement that was not requested.
The per-pass warm-up progress goes to stderr, the tables to stdout, so `> results.md` keeps just the tables.

## What the numbers are

The first table is `llama-bench`'s workload, one row per test.

| test | what is timed |
|------|---------------|
| `pp512` | 512 synthetic tokens ingested in chunks of 512 (`llama-bench -ub 512`), one logits projection for the last token, context sized to the work |
| `tg128` | 128 single-token decodes from an empty state, positions 0 to 127, logits projected every step and never argmaxed, as `llama-bench` feeds back `rand() % n_vocab` |

One state per test, reset before every warm-up pass and every timed repetition, so no allocation or page fault sits inside a timed region.
Warm-up runs whole passes until throughput settles within 3% over a window of three (at least `-w`, at most 30), because a JIT needs more than the one pass native code needs.
`-d N` keeps `N` resident prefix tokens, prepared outside the timing, as `llama-bench -d`.

The second table is what the engine adds on top of the forward pass, measured through `ChatEngine` on a real prompt: state allocation, time to first token from cold, a prompt-cache hit against a full re-prefill, projected-media cold versus warm latency (`--media`), and peak RSS.

**A draft head is never part of a throughput number.**
`pp512` and `tg128` drive the model at the state and batch boundary and know nothing about speculation.
The `mtp` cell is a separate measurement of draft acceptance and net decode speed-up, and it runs only when the draft head is attached with `--with speculation=<mtp.gguf>`; without one it reads `no draft head`.

## What decides the numbers

**The compiler.**
A native image and the GraalVM JIT compile the Vector API kernels fully.
C2 runs the byte-unpacking quant kernels largely through the un-intrinsified per-lane fallback, and the engine says so at startup with the `SLOW_JIT` cliff and routes around it where it can.
On a GraalVM JDK the Graal JIT is the default; `-XX:-UseJVMCICompiler` selects C2.
The header line tells you which one ran; do not compare a C2 number with a Graal number.

**The backend.**
Matmuls try the jam backends in order - `native` (the C kernels in `libjam`), `vector` (the Java Vector API kernels), `scalar` (autovectorized Java) - and fall to the engine's own floor when none applies.
Prefill takes the first backend that has a kernel for the weight type.
Decode takes the native gemv only where it wins, which on x86 means a narrow pool (the crossover is between 4 and 16 threads by quant); an explicit `-Djinfer.q4.nativeDecode=true|false` (also `q8`, `mxfp4`, `kq`) forces either way.
`-Djam.native.disabled=true` (or `vector`, `scalar`) removes a backend before it is probed, which is how the Java kernels are A/B-tested against the native ones; the header shows what is left.

**Threads.**
The default is one thread per physical core - P-cores only on Apple Silicon, as `llama-bench` does - and `-t N` forces the same count for both tests and every backend.
Decode is memory-bound at high thread counts, so more threads past the crossover do not help it and can hurt.

**The warnings.**
The engine reports a performance cliff once per run, on stderr, as `perf cliff [NAME]: ...`.
A reported number must come from a run with no cliff, or say which one it carried.

| cliff | it means | do this |
|-------|----------|---------|
| `SLOW_JIT` | this JIT compiles the Vector API kernels conservatively (C2) | run on the GraalVM JIT or the native image |
| `JAM_ABSENT` | no jam backend on the classpath, prefill on the pure-Java path | use the shaded jar or the image; a hand-built classpath needs `jam-native` or `jam-vector` |
| `JAM_DECLINE` | the native kernels declined a matmul shape they were offered | a mismatched `libjam`; rebuild the natives, and report it if it persists |
| `NATIVE_ACCESS_RESTRICTED` | vector accesses keep checks native access would lift | `--enable-native-access=ALL-UNNAMED`; the shaded jar's manifest sets it |
| `MAMBA2_SCALAR`, `GDN_SCALAR` | this model's recurrent scan geometry has no vector kernel yet | a property of the model on this build, not of your setup; report the cliff with the number |

## Doing it properly

- Run on an idle machine, plugged in, with nothing else scheduled; decode is memory-bound and any background load shows in `tg128`.
- One process per configuration.
  Several knobs (`convTile`, `vectorJit`, the thread budget) are read once at class initialization; changing them for a second model in the same process measures the first model's settings.
- Same thread count on both sides of a comparison, and say what it is.
- ABBA: when comparing two engines or two configurations, run A, B, B, A so thermal drift and background load hit both sides equally.
- At least `-r 5`; the standard deviation is part of the number.
- Never quote a cold single run - the first passes of a JVM are the JIT, not the engine - and never quote a number from another machine.
  Re-measure on yours before writing anything down.

The llama.cpp side of a comparison, same lengths, same repetitions, same threads, a llama.cpp built for the same machine:

```bash
llama-bench -m model.gguf -p 512 -n 128 -r 5 -t 16
```

`llama-bench`'s `-b 2048` only caps tokens per decode call; ggml still computes 512-row micro-batches, which is what `pp512` here does too.

A report is the header line, the tables, and the machine, nothing paraphrased:

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
| `-m, --model <path>` | GGUF to benchmark, repeatable; any architecture on the classpath |
| `-p, --n-prompt <N>` | prefill tokens, default 512; `0` skips `pp` |
| `-n, --n-gen <N>` | decode tokens, default 128; `0` skips `tg` |
| `-d, --n-depth <N>` | resident prefix tokens, prepared outside the timing, default 0 |
| `-r, --repetitions <N>` | timed repetitions, default 5 |
| `-w, --warmup <N>` | minimum warm-up passes, default 2; then adaptive until throughput settles |
| `--no-warmup` | no warm-up at all; for native code only |
| `--no-capabilities` | skip the second table |
| `-t, --threads <N>` | thread budget for both tests and every backend; default: one per physical core |
| `--ctx <N>` | context for both tests; default: `p` for `pp`, `n` for `tg`, as `llama-bench` |
| `--with <capability>=<path>` | attach a companion, repeatable: `media=<mmproj.gguf>`, `speculation=<mtp.gguf>` |
| `--media <image>` | also measure projected-media cold and warm latency; needs `--with media=` |

Vision, with the projector attached and an image measured:

```bash
bin/jinfer-bench -m gemma-4-12b-it-Q8_0.gguf -p 512 -n 128 -r 5 -w 2 \
  --with media=mmproj-F32.gguf --media cat.png
```

### EmbedBench

`EmbedBench` measures the packed-embedding path (`EmbeddingModel.embedAll`): many variable-length sequences packed into segmented forwards over one KV context, every pooled vector streamed out.
The workload is deterministic - ragged lengths by multiplicative hash, greedy filler tokens.

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
`llama-bench --embeddings 1` uses one flat 512-token prompt, so only `tok/s` is comparable.

### Microbenchmarks

Loops over engine kernels, no model loaded; run them with the same `-cp` as `EmbedBench`.

- `SpinProbe [iters]`: the cost of one empty `Parallel` region - the dispatch and barrier latency every parallel region of a decode token pays.
- `ConvPeak [census.log]`: `Convolutions.conv1dRows` throughput across the register-tile shapes; with a shape census from `-Djinfer.convProfile=true` on a real synthesis it measures the shapes that model ran, weighted by FLOPs.
  The tile shape is a constant, so one process measures one shape: run three times with `-Djinfer.convTile=auto|4x2|4x4`.
- `ConvParity`: whether the tile shape changes the numbers; run once per `-Djinfer.convTile` value and `diff` the outputs.
- `bench/DeltaNetParity.java` in `jinfer/`: standalone, no dependencies, chunked gated DeltaNet against the sequential recurrence (`javac bench/DeltaNetParity.java -d /tmp/dn && java -cp /tmp/dn DeltaNetParity`).

Model-level probes live as tagged JUnit benches next to their models - `MtpBench` (jinfer-gemma4), `PrefillBench`, `ScoringBench`, `GrammarCostProbe`, `NmtProbe` (jinfer-langchain4j) - and run by name: `mvn -pl jinfer/jinfer-gemma4 test -Dtest=MtpBench -Dsurefire.excludedGroups=`.

### Measurement instruments

These change what is compiled or recorded and are for finding a regression, not for a reported number; each carries its own instructions.

- [`hotspot_compile_commands`](../hotspot_compile_commands): C2 inlining hints for the hot Vector API helpers, to tell an inlining regression from a kernel regression under C2. The Graal JIT does not read it.
- `-Djinfer.convTile=auto|4x2|4x4` and `-Djinfer.convProfile=true`: the convolution register tile and its shape census, documented in `jinfer/pom.xml` and `Convolutions.TILE_CODE`.
- `-Djinfer.vectorJit=auto|fast|slow`: override the compiler detection behind `SLOW_JIT`, for a JIT the detection misjudges.
- JFR (`-XX:StartFlightRecording`): start it after warm-up, or filter samples by time; a recording that starts with the JVM attributes the interpreter and C1 to the kernels.

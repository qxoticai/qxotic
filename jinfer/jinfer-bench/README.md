# Jinfer Bench

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

Benchmark harness for the [Jinfer](../README.md) inference engine. This page documents the workloads
and commands used for [`BENCHMARKS.md`](../../BENCHMARKS.md).

## Build

From the repository root, install the reactor and build the benchmark classpath:

```bash
mvn install -DskipTests
mvn -q -pl jinfer/jinfer-bench dependency:build-classpath -Dmdep.outputFile=target/cp.txt
```

Every harness below uses the same JVM flags and the inlining hints in
[`hotspot_compile_commands`](../hotspot_compile_commands). The hints force hot Vector API helpers to inline. Without
them, some JIT configurations leave a helper out of line, causing vector boxing and slower prefill.
Using the file keeps runs comparable across JVMs:

```bash
BENCH_FLAGS="--add-modules jdk.incubator.vector,jdk.httpserver \
  --enable-native-access=ALL-UNNAMED \
  -XX:CompileCommandFile=jinfer/hotspot_compile_commands \
  -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0"
CP="jinfer/jinfer-bench/target/classes:$(< jinfer/jinfer-bench/target/cp.txt)"
```

## JinferBench: prefill, decode and engine capabilities

`JinferBench` drives every model through the generic loader (`Models.load`; any architecture on
the classpath runs, no per-model code) and reports two kinds of results:

- **Prefill and decode tests**, matching the `llama-bench` workload: `pp512` processes 512
  synthetic tokens in chunks of 512 with the context sized to the work; `tg128` decodes 128
  single tokens from an empty state. Logits are projected every step but never argmaxed, like
  llama-bench's `rand() % n_vocab` feedback. One state per test, `reset()` before every warmup
  pass and timed rep, so no allocation happens inside the timed region.
- **Capability measurements** through `ChatEngine`: state allocation, cold TTFT, prompt-cache hit vs
  full re-prefill, MTP draft acceptance and decode speedup (when the model carries a draft head),
  projected-media cold vs warm latency (with `--media`), and peak RSS (`VmHWM`).

```bash
java $BENCH_FLAGS -cp "$CP" com.qxotic.jinfer.bench.JinferBench \
  -m model.gguf -p 512 -n 128 -r 5 -w 2
```

| Option | Meaning |
|--------|---------|
| `-m, --model <path>` | model to benchmark (repeatable) |
| `-p, --n-prompt <N>` | prefill tokens, default 512; `0` skips prefill |
| `-n, --n-gen <N>` | decode tokens, default 128; `0` skips decode |
| `-r, --repetitions <N>` | timed reps, default 5 |
| `-w, --warmup <N>` | minimum warmup passes; then adaptive until throughput settles within 3% (max 30) |
| `--no-warmup` | skip warmup entirely (native code only) |
| `-t, --threads <N>` | worker count for both tests, default physical cores; sets `jinfer.computeThreads`, `jinfer.decodeThreads` and `jam.threads` |
| `--ctx <N>` | override context for both tests (default: `p` for pp, `n` for tg, as llama-bench) |
| `--with media=<mmproj.gguf>` | attach a companion (repeatable; CLI convention) |
| `--media <path>` | also measure projected-media cold/warm latency (needs a vision projector) |

Warmup is adaptive (full-test passes until throughput settles) instead of llama-bench's single
pass, because the JIT needs several; tokens are synthetic in-range ids; loading is direct and never
timed. See the class Javadoc for the full list of differences.

Vision run using the configuration from `benchmarks-vision.csv`:

```bash
java $BENCH_FLAGS -cp "$CP" com.qxotic.jinfer.bench.JinferBench \
  -m gemma-4-12b-it-Q8_0.gguf -p 512 -n 128 -r 5 -w 2 \
  --with media=mmproj-F32.gguf --media cat.png
```

The llama.cpp side of a parity table, with the same lengths and reps:

```bash
llama-bench -m model.gguf -p 512 -n 128 -r 5 -t 16
```

## EmbedBench: ragged batched-embedding throughput

`EmbedBench` measures the packed-embedding path (`EmbeddingModel.embedAll`): many variable-length
sequences packed into segmented forwards over one KV context, each pooled vector streamed out. The
workload is deterministic (ragged lengths by multiplicative hash, greedy filler tokens).

```bash
java $BENCH_FLAGS -cp "$CP" com.qxotic.jinfer.bench.EmbedBench \
  -m embedder.gguf -s 256 --minlen 8 --maxlen 64 -b 512 -r 5 -w 3
```

| Option | Meaning |
|--------|---------|
| `-m, --model <path>` | embedding checkpoint (bidirectional or causal; the port declares pooling) |
| `-s, --sequences <N>` | number of packed sequences, default 256 |
| `--minlen, --maxlen <N>` | ragged length range, default 8 to 64 |
| `-b, --batch <N>` | per-chunk forward width / `batchCapacity`, default 512 |
| `-r, --repetitions <N>` | timed reps, default 5 |
| `-w, --warmup <N>` | minimum warmup passes, default 3 (then adaptive within 3%) |
| `-t, --threads <N>` | compute threads |

Reports both `tok/s` (total packed tokens) and `seq/s`. The llama.cpp side (`llama-bench
--embeddings 1`) uses one flat 512-token prompt, so only `tok/s` is comparable.

## Microbenchmarks

Small loops over engine-level kernels, without loading a model:

- **`SpinProbe [iters]`**: cost of a `SpinPool` dispatch and barrier with an empty region. This is
  the dispatch latency paid by each parallel decode region. It runs on the decode pool, so it
  uses the real spin path. `java $BENCH_FLAGS -cp "$CP" com.qxotic.jinfer.bench.SpinProbe`.
- **`ConvPeak [census.log]`**: `Convolutions.conv1dRows` throughput across the register-tile
  shapes. With a shape census from `-Djinfer.convProfile=true` on a real synthesis it measures
  exactly the shapes that model ran, weighted by FLOPs; without one it falls back to a generic
  vocoder ladder. The tile shape is a `static final` (`Convolutions.TILE_CODE`), so one JVM
  measures one shape. Sweep by running three times with `-Djinfer.convTile=auto|4x2|4x4`.
- **`ConvParity`**: checks whether the tile shape changes the numbers. Run once per
  `-Djinfer.convTile` value and diff the outputs; identical digests mean the tiles are bit-for-bit
  equivalent. `diff /tmp/parity.auto /tmp/parity.4x2 && diff /tmp/parity.auto /tmp/parity.4x4`.
- **`bench/DeltaNetParity.java`**: standalone, no dependencies. It compares chunked gated DeltaNet
  with the sequential recurrence. From `jinfer/`, run `javac bench/DeltaNetParity.java -d /tmp/dn
  && java -cp /tmp/dn DeltaNetParity`.

Further model-level probes live as tagged JUnit benches next to their models: `MtpBench`
(jinfer-gemma4), `PrefillBench`/`ScoringBench`/`GrammarCostProbe`/`NmtProbe` (jinfer-langchain4j).
Run them by name: `mvn test -pl jinfer-gemma4 -Dtest=MtpBench -Dsurefire.excludedGroups=`.

## Reproducing results

The rules behind `BENCHMARKS.md`:

- **Match the worker counts.** Pass the same physical-core count to pp and tg with `-t N`. This
  configures Jinfer and JAM together. A provider-specific `JAM_<PROVIDER>_THREADS` setting still
  takes precedence. Record the resolved counts printed by the harness.
- **Same JVM flags.** The `BENCH_FLAGS` above, verbatim. The `hotspot_compile_commands` hints and the
  `VECTOR_ACCESS_OOB_CHECK=0` flag are part of the measurement.
- **ABBA reruns.** When comparing two engines or two configurations, alternate the order (A-B-B-A)
  so thermal drift and background load hit both sides equally. Use a fresh JVM per run, because
  several knobs (`convTile`, `TILE_CODE`, warmup windows) are set at class-init time.
- **Report the context.** Include the mean and standard deviation over timed repetitions, host,
  thread count and commit.
- **Do not compare across machines.** Results depend on the machine. Re-run on your own machine
  before quoting anything.

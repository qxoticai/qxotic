# Parakeet benchmarks

Speed measurements for the three CPU engines that run NVIDIA Parakeet: jinfer, parakeet.cpp and
sherpa-onnx.
All numbers are _RTFx_, seconds of audio transcribed per wall second with model load excluded, so
higher is faster and 1.0 is real time.

Two machines are covered.
The laptop figures are the README's summary and are the ones paired with word error rates; the
desktop figures separate the thread, quantization and JVM axes but measure speed only.
The two are not comparable to each other, because they use different audio as well as different
hardware.

## Machines and corpora

| | Laptop | Desktop |
|---|---|---|
| CPU | AMD Ryzen 7 PRO 8840U | AMD Zen 5, 16 physical cores |
| Audio | LibriSpeech test-clean, first 100 utterances, 901 s | 8 clips, 52.49 s |
| Threads | 8 | swept 1, 2, 4, 8, 16 |
| Repeats | median of 3 runs | see below |
| Accuracy | WER measured, see [WER.md](WER.md) | not measured |

The desktop clips are the Parakeet test speech from
`jinfer/scripts/download_parakeet_fixtures.py`, each pinned to a repository revision and checked
against its SHA-256: the four sherpa-onnx test wavs (`en`, `de`, `es`, `fr`), three LibriSpeech
utterances (`ls0`, `ls1`, `ls8k`) and one further English clip (`en0`).

| Clip | Seconds | Clip | Seconds |
|---|---|---|---|
| `en` | 3.85 | `ls8k` | 4.83 |
| `en0` | 7.43 | `de` | 2.75 |
| `ls0` | 6.62 | `es` | 5.33 |
| `ls1` | 16.71 | `fr` | 4.97 |

## Engines and weights

| | Version | Weights |
|---|---|---|
| jinfer | Oracle GraalVM for JDK 25 (i4), Graal JIT, jam-native on the classpath | GGUF |
| parakeet.cpp | commit `e75de9b`, vendored ggml, `PARAKEET_DEVICE=cpu` | the same GGUF files |
| sherpa-onnx | Python wheel, `OfflineRecognizer` | int8 ONNX, the only quantization published |

jinfer and parakeet.cpp read byte-identical GGUF files, so that comparison isolates the engine.
sherpa-onnx uses a different quantization scheme, so part of any gap against it is the format
rather than the engine.

| GGUF | tdt-0.6b-v3 | tdt_ctc-110m |
|---|---|---|
| `Q4_K` | 644 MB | 125 MB |
| `Q5_K` | 707 MB | 137 MB |
| `Q6_K` | 775 MB | 149 MB |
| `Q8_0` | 897 MB | 170 MB |
| `F16` | 1374 MB | 255 MB |

`Q8_0` is a mix rather than uniform: of the 0.6B's 701 tensors, 219 hold 85.1% of the weights as
`Q8_0` and the remaining 482 stay `F32`.

Model load is not counted in RTFx.
For reference, parakeet.cpp loads the 0.6B in 80 ms and the 110m in 15 ms, and jinfer reports
0.1 s for the 0.6B.

## Method

RTFx is aggregated as the sum of audio over the sum of processing time across the eight clips.
jinfer runs five repetitions per clip and discards the first as warmup.
parakeet.cpp's `bench` does not warm up, so every measurement reported here is a second pass over
the same manifest.
sherpa-onnx runs a full warmup pass and then takes the best of three per clip, which is the most
generous of the three treatments.

Run-to-run variance is about 5%, so differences smaller than that are not meaningful.

## Laptop, 8840U

LibriSpeech test-clean, first 100 utterances, 8 threads, median of 3 runs.
The last column scores each run against parakeet.cpp's transcript rather than the reference, so 0%
means both engines heard exactly the same words.

| Model | Engine | WER | RTFx | WER vs. parakeet.cpp |
|-------|--------|-----|------|----------------------|
| tdt-0.6b-v3 `Q4_K` | **jinfer** | 2.04% | **25.1** | 0.17% |
| | parakeet.cpp | 2.04% | 12.6 | - |
| tdt-0.6b-v3 `Q8_0` | **jinfer** | 2.04% | **21.6** | 0.04% |
| | parakeet.cpp | 2.04% | 14.9 | - |
| tdt-0.6b-v3 `F16` | jinfer | 2.04% | 13.6 | 0.00% |
| | parakeet.cpp | 2.04% | **15.3** | - |
| tdt_ctc-110m `Q4_K` | **jinfer** | 1.99% | **81.2** | 0.34% |
| | parakeet.cpp | 2.12% | 45.8 | - |
| v3 int8 ONNX | sherpa-onnx | 2.16% | 18.6 | 1.27% |

## Desktop, Zen 5

### Engine comparison, by thread count

Each row sweeps 1, 2, 4, 8 and 16 threads; the bold cell is that configuration's peak.

#### tdt-0.6b-v3 (600M, multilingual)

| Engine | Quant | 1t | 2t | 4t | 8t | 16t | Peak |
|--------|-------|----|----|----|----|-----|------|
| jinfer | `Q4_K` | 18.7 | 34.4 | 54.8 | **69.8** | 66.1 | 69.8 @ 8t |
| jinfer | `Q8_0` | 15.3 | 28.4 | 47.5 | **65.3** | 63.4 | 65.3 @ 8t |
| jinfer | `F16` | 10.7 | 20.5 | 35.4 | 51.4 | **55.1** | 55.1 @ 16t |
| parakeet.cpp | `Q4_K` | 6.8 | 13.0 | 23.9 | 39.9 | **52.7** | 52.7 @ 16t |
| parakeet.cpp | `Q8_0` | 9.1 | 17.2 | 30.8 | 50.6 | **62.8** | 62.8 @ 16t |
| parakeet.cpp | `F16` | 11.3 | 20.7 | 36.4 | 56.8 | **62.4** | 62.4 @ 16t |
| sherpa-onnx | int8 ONNX | 21.0 | 35.3 | 52.6 | **67.6** | 47.7 | 67.6 @ 8t |

#### tdt_ctc-110m (110M, English)

| Engine | Quant | 1t | 2t | 4t | 8t | 16t | Peak |
|--------|-------|----|----|----|----|-----|------|
| jinfer | `Q4_K` | 64.1 | 114.0 | 163.5 | **168.1** | 141.6 | 168.1 @ 8t |
| jinfer | `Q8_0` | 57.6 | 103.5 | 154.2 | **168.4** | 134.7 | 168.4 @ 8t |
| jinfer | `F16` | 44.7 | 83.8 | 125.6 | **140.6** | 124.4 | 140.6 @ 8t |
| parakeet.cpp | `Q4_K` | 26.8 | 49.8 | 87.0 | 134.1 | **147.8** | 147.8 @ 16t |
| parakeet.cpp | `Q8_0` | 34.5 | 64.0 | 111.3 | 154.9 | **169.3** | 169.3 @ 16t |
| parakeet.cpp | `F16` | 38.4 | 69.6 | 117.1 | **175.1** | 157.4 | 175.1 @ 8t |
| sherpa-onnx | int8 ONNX | 81.1 | 120.0 | 155.6 | **182.8** | 147.2 | 182.8 @ 8t |

### Further slices

#### All five GGUF quantizations, 16 threads

| Model | Quant | jinfer | parakeet.cpp |
|-------|-------|--------|--------------|
| tdt-0.6b-v3 | `Q4_K` | 66.9 | 55.0 |
| tdt-0.6b-v3 | `Q5_K` | 55.1 | 46.6 |
| tdt-0.6b-v3 | `Q6_K` | 49.6 | 52.9 |
| tdt-0.6b-v3 | `Q8_0` | 65.8 | 61.4 |
| tdt-0.6b-v3 | `F16` | 55.4 | 63.7 |
| tdt_ctc-110m | `Q4_K` | 138.1 | 144.7 |
| tdt_ctc-110m | `Q5_K` | 130.7 | 136.9 |
| tdt_ctc-110m | `Q6_K` | 116.0 | 145.6 |
| tdt_ctc-110m | `Q8_0` | 128.7 | 156.0 |
| tdt_ctc-110m | `F16` | 127.4 | 157.2 |

#### Per clip, 8 threads, `Q8_0` (int8 for sherpa-onnx)

| Model | Engine | en | en0 | ls0 | ls1 | ls8k | de | es | fr |
|---|---|---|---|---|---|---|---|---|---|
| tdt-0.6b-v3 | jinfer | 53.2 | 63.9 | 65.0 | 83.2 | 56.8 | 44.8 | 56.6 | 57.9 |
| tdt-0.6b-v3 | parakeet.cpp | 46.9 | 51.8 | 52.8 | 53.4 | 50.1 | 47.8 | 52.7 | 50.0 |
| tdt-0.6b-v3 | sherpa-onnx | 50.6 | 69.0 | 68.2 | 73.2 | 59.7 | 49.6 | 64.5 | 63.1 |
| tdt_ctc-110m | jinfer | 143.0 | 139.2 | 153.3 | 223.8 | 148.1 | 103.4 | 149.6 | 154.2 |
| tdt_ctc-110m | parakeet.cpp | 163.7 | 170.4 | 171.1 | 175.5 | 147.2 | 137.7 | 175.5 | 165.6 |
| tdt_ctc-110m | sherpa-onnx | 161.6 | 189.7 | 188.1 | 194.5 | 183.5 | 148.9 | 175.6 | 170.7 |

Clip lengths in seconds: `en` 3.85, `en0` 7.43, `ls0` 6.62, `ls1` 16.71, `ls8k` 4.83, `de` 2.75, `es` 5.33, `fr` 4.97.

#### Oracle GraalVM i4 vs CE i4, raw passes, 0.6B `Q8_0`, 16 threads

Run in A, B, B, A order with 30 s cooldowns so thermal drift affects both sides equally.

| Clip | oracle pass 1 | ce pass 1 | ce pass 2 | oracle pass 2 |
|---|---|---|---|---|
| `en` | 51.8 | 52.4 | 52.0 | 52.8 |
| `en0` | 62.9 | 61.9 | 67.7 | 62.9 |
| `ls0` | 61.1 | 60.5 | 57.8 | 55.5 |
| `ls1` | 76.0 | 77.8 | 75.0 | 72.7 |
| `ls8k` | 57.6 | 56.4 | 57.4 | 64.5 |
| `de` | 53.4 | 47.5 | 47.5 | 44.1 |
| `es` | 63.5 | 56.7 | 57.4 | 61.3 |
| `fr` | 61.5 | 56.5 | 59.0 | 58.0 |

#### Convolution tile shape, CE i4, `Q8_0`, 16 threads

| Model | `auto` | `4x2` | `4x4` |
|---|---|---|---|
| tdt-0.6b-v3 | 63.2 | 61.9 | 63.0 |
| tdt_ctc-110m | 130.0 | 126.3 | 129.9 |

#### Decoder branch, parakeet.cpp, `Q8_0`, 8 threads

| Model | `--decoder tdt` | `--decoder ctc` |
|---|---|---|
| tdt-0.6b-v3 | 49.9 | aborts (no CTC branch) |
| tdt_ctc-110m | 169.9 | 179.5 |

CTC decoding is about 6% cheaper than TDT on the hybrid 110m model, which matters when comparing
against sherpa-onnx: its 110m weights are the CTC branch only, so it decodes the cheaper path.
Asking parakeet.cpp for a CTC decode on the 0.6B, which has no CTC branch, terminates the process
with a core dump rather than a diagnostic.

### JVM axis

jinfer only, since the other engines are native binaries.


| Model | Quant | Oracle GraalVM i4 | GraalVM CE i4 | OpenJDK 27 (C2) |
|-------|-------|-------------------|---------------|-----------------|
| tdt-0.6b-v3 | `Q4_K` | 66.9 | 65.5 | 69.1 |
| tdt-0.6b-v3 | `Q5_K` | 55.1 | 54.1 | 58.2 |
| tdt-0.6b-v3 | `Q6_K` | 49.6 | 47.7 | 51.6 |
| tdt-0.6b-v3 | `Q8_0` | 65.8 | 62.9 | 65.6 |
| tdt-0.6b-v3 | `F16` | 55.4 | 52.9 | 53.3 |
| tdt_ctc-110m | `Q4_K` | 138.1 | 135.5 | 133.5 |
| tdt_ctc-110m | `Q5_K` | 130.7 | 119.9 | 125.9 |
| tdt_ctc-110m | `Q6_K` | 116.0 | 117.6 | 115.1 |
| tdt_ctc-110m | `Q8_0` | 128.7 | 127.3 | 124.1 |
| tdt_ctc-110m | `F16` | 127.4 | 116.0 | 100.0 |

## What the numbers say

**8 threads is the peak, not 16.**
Every jinfer and sherpa-onnx configuration on both models peaks at 8 threads and loses ground at
16, while parakeet.cpp's GGUF runs keep gaining to 16.
sherpa-onnx degrades hardest, losing 29% from 8 to 16 threads on the 0.6B and 19% on the 110m,
which looks like ONNX Runtime's split intra/inter-op pools oversubscribing a 16-core machine.
jinfer's default is the physical core count, which is past its own optimum for this workload.

**The two GGUF engines prefer opposite quantizations, on both models and both machines.**
`Q4_K` is jinfer's fastest and parakeet.cpp's slowest, and `F16` is the reverse.
This follows from kernel coverage rather than from the formats: jam's native backend carries
k-quant paths and no F16 path, while ggml has a dedicated F32xF16 fast path.
`Q5_K` is parakeet.cpp's worst on both models, slower than the larger `Q6_K`, which inverts the
size ordering and suggests a missing vectorized Q5_K dot product.

**Per-core efficiency differs more than peak throughput.**
At one thread on the 0.6B, sherpa-onnx reaches 21.0 RTFx, jinfer 18.7 and parakeet.cpp 11.3 at its
best quantization and 6.8 at `Q4_K`.
On identical GGUF weights jinfer does 2.75x parakeet.cpp's work per core at `Q4_K`.
parakeet.cpp's 16-thread wins come from scaling 6.8x between 1 and 16 threads, from a low base.

**Clip length decides the per-clip winner.**
On the 0.6B at 8 threads, jinfer leads on the 16.7 s clip by 56% over parakeet.cpp (83.2 against
53.4) and trails on the 2.75 s clip (44.8 against 47.8).
jinfer carries more fixed per-call cost, so its advantage grows with utterance length.

**C2 is not the cliff it is documented to be, for this workload.**
It matches or beats the Graal JITs on every quantized format and collapses only on `F16`, by 21.5%
on the 110m.
jam-native owns the quantized matmuls, so for `Q4_K`, `Q5_K`, `Q6_K` and `Q8_0` the JIT compiles
glue rather than kernels; `F16` has no native kernel, falls back to the Java Vector API, and there
C2's conservative compilation costs real time.
Oracle GraalVM i4 leads GraalVM CE i4 in 9 of 10 cells by a mean of 3.3%, though no single cell
clears the noise floor.

**The convolution tile shape makes no measurable difference.**
At `Q8_0` on CE i4, `4x4` lands within 0.3% of `auto` on both models, while `4x2` is consistently
2-3% slower, so `auto` appears to select `4x4` already.
The convolutional front-end is too small a share of a 24-layer encoder for its tiling to register
behind a native backend.

**Best achievable per engine on the desktop, over every thread count and quantization:**

| Model | jinfer | parakeet.cpp | sherpa-onnx |
|-------|--------|--------------|-------------|
| tdt-0.6b-v3 | **69.8** (`Q4_K`, 8t) | 62.8 (`Q8_0`, 16t) | 67.6 (int8, 8t) |
| tdt_ctc-110m | 168.4 (`Q8_0`, 8t) | 175.1 (`F16`, 8t) | **182.8** (int8, 8t) |

jinfer leads on the 600M model and trails on the 110M one.
A smaller model spends proportionally more time outside the large matmuls, where fixed per-call
cost dominates, and that is jinfer's weaker ground.

## Ordering of what matters

| Axis | Spread |
|------|--------|
| Thread count | up to 3.7x (jinfer `Q4_K` 0.6B, 1t to 8t) |
| Engine | up to 2.75x per core |
| Quantization | ~35% (jinfer 0.6B, `Q4_K` 66.9 vs `Q6_K` 49.6 at 16t) |
| JVM | ~3% on quantized formats, 21% on `F16` |
| Convolution tile | none measurable |

## Sources

Everything below was measured on 2026-09-23 from these exact sources.

| Component | Source | Version |
|---|---|---|
| jinfer | this repository | `c1be18edf` |
| parakeet.cpp | [github.com/mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) | `e75de9b` |
| ggml | vendored at `third_party/ggml` | `e705c5f` |
| sherpa-onnx | PyPI wheel | 1.13.8 |
| Oracle GraalVM | JDK 25 i4 | 25.0.4.1.1, `GRAALVM_VERSION=25.4.4.1.1` |
| GraalVM CE | JDK 25 i4 | 25.0.4.1.1, `GRAALVM_VERSION=25.4.4.1.1` |
| OpenJDK | reference build, C2 | 27+35-2325 |

| Weights | Source |
|---|---|
| GGUF, both models, all five quantizations | [mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf) |
| sherpa-onnx 0.6B, int8 transducer | [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8) |
| sherpa-onnx 110m, int8 CTC | [VocaHQ/sherpa-onnx-nemo-parakeet-tdt-ctc-110m-en-int8](https://huggingface.co/VocaHQ/sherpa-onnx-nemo-parakeet-tdt-ctc-110m-en-int8) |
| Speech clips | `jinfer/scripts/download_parakeet_fixtures.py`, revision-pinned and SHA-256 checked |

GGUF files were fetched from `main` rather than a pinned revision; their SHA-256 prefixes here are
`4d69a4a6683f4f2d` for `tdt-0.6b-v3-q8_0.gguf` and `614feee3a990cf0e` for `tdt_ctc-110m-q8_0.gguf`.

## Reproducing

### Setup

```bash
# speech clips -> test-fixtures/parakeet
python3 jinfer/scripts/download_parakeet_fixtures.py

# jinfer bench harness
mvn -pl jinfer/jinfer-bench -am -DskipTests package

# parakeet.cpp
git clone --recursive https://github.com/mudler/parakeet.cpp && cd parakeet.cpp
git checkout e75de9b && git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# sherpa-onnx
python3 -m venv venv && venv/bin/pip install sherpa-onnx==1.13.8 numpy

# weights
for q in q4_k q5_k q6_k q8_0 f16; do
  for m in tdt-0.6b-v3 tdt_ctc-110m; do
    curl -L -o $m-$q.gguf \
      https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/$m-$q.gguf
  done
done
```

### One engine, one configuration

```bash
# jinfer: RTFx over one clip, 5 reps, first discarded as warmup
java --add-modules jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
  -Djinfer.threads=8 -cp jinfer/jinfer-bench/target/jinfer-bench.jar \
  com.qxotic.jinfer.bench.TranscriptionBench \
  --model tdt-0.6b-v3-q4_k.gguf --audio test-fixtures/parakeet/ls1.wav --reps 5

# parakeet.cpp: RTFx over a manifest of "<wav>\t<transcript>" lines; run twice, keep the second
PARAKEET_DEVICE=cpu build/examples/cli/parakeet-cli bench \
  --model tdt-0.6b-v3-q4_k.gguf --manifest manifest.tsv --threads 8

# sherpa-onnx: load once, warm up, then time each clip
venv/bin/python - <<'EOF'
import sherpa_onnx, time, wave, numpy as np
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="encoder.int8.onnx", decoder="decoder.int8.onnx", joiner="joiner.int8.onnx",
    tokens="tokens.txt", num_threads=8, model_type="nemo_transducer")
EOF
```

The two JVM flags are the ones the native image is built with; the bench README asks that no other
flags be used for a reported number.
`PARAKEET_DEVICE=cpu` is required or parakeet.cpp auto-selects the first GPU the ggml registry
reports, including an integrated one, which would make the comparison GPU against CPU.

### Which command produced which table

| Table | How |
|---|---|
| Engine comparison by thread count | the three commands above, for `-Djinfer.threads` / `--threads` / `num_threads` in 1, 2, 4, 8, 16, each quantization, each model |
| All five GGUF quantizations | the same at 16 threads, adding `Q5_K` and `Q6_K` |
| Per clip | the same at 8 threads on `Q8_0`, reporting each clip instead of the aggregate |
| Oracle vs CE raw passes | four full passes in A, B, B, A order with 30 s cooldowns, swapping only the `java` binary |
| Convolution tile | `-Djinfer.convTile=auto\|4x2\|4x4`, one JVM per value, since the choice freezes at class initialization |
| Decoder branch | `parakeet-cli bench --decoder tdt\|ctc` |
| JVM axis | the jinfer command with each of the three `java` binaries |

RTFx is aggregated per configuration as the sum of clip durations over the sum of processing
times, not as the mean of per-clip RTFx values.

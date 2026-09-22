# Parakeet accuracy and speed

Word error rate and RTFx on LibriSpeech test-clean, measured as
[parakeet.cpp](https://github.com/mudler/parakeet.cpp) measures them, so the numbers compare: its
first 100 utterances in sorted order (speakers 1089 and 1188, 901 s of audio), and its
normalization (NFKC, lowercase, punctuation to spaces, collapsed whitespace) before a word-level
edit distance over the reference length.

RTFx is seconds of audio per wall second, model load excluded. Agreement is WER against another
engine's transcript of the same audio: 0% means both heard the same words, which separates a port
bug from a model difference.

Everything below runs from the repository root.

## 1. Corpus

LibriSpeech test-clean is 346 MB; only two speakers are needed.

```bash
mkdir -p test-fixtures/librispeech
curl -o test-fixtures/librispeech/test-clean.tar.gz \
  https://www.openslr.org/resources/12/test-clean.tar.gz
tar xzf test-fixtures/librispeech/test-clean.tar.gz -C test-fixtures/librispeech \
  LibriSpeech/test-clean/1089 LibriSpeech/test-clean/1188
```

Those two speakers hold 109 utterances; the first 100 in sorted order are the benchmark set, which
is what `--limit 100` takes. parakeet.cpp and sherpa-onnx read 16 kHz WAV and a
`<path>\t<reference>` manifest instead, built here (needs `ffmpeg`):

```bash
python3 - <<'EOF'
import pathlib, subprocess
root = pathlib.Path("test-fixtures/librispeech/LibriSpeech/test-clean")
wavs = pathlib.Path("test-fixtures/librispeech/wav"); wavs.mkdir(exist_ok=True)
said = {}
for transcript in sorted(root.rglob("*.trans.txt")):
    for line in transcript.read_text().splitlines():
        name, reference = line.split(" ", 1)
        said[name] = (transcript.parent / f"{name}.flac", reference)
rows = []
for name in sorted(said)[:100]:
    flac, reference = said[name]
    wav = wavs / f"{name}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(flac),
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)
    rows.append(f"{wav}\t{reference}")
pathlib.Path("test-fixtures/librispeech/manifest.tsv").write_text("\n".join(rows) + "\n")
print(len(rows), "utterances")
EOF
```

With a parakeet.cpp checkout at hand, the set matches its manifest exactly:

```bash
diff <(cut -f1 test-fixtures/librispeech/manifest.tsv | xargs -n1 basename | sed 's/.wav//') \
     <(cut -f1 ../parakeet.cpp/benchmarks/librispeech_manifest.tsv | xargs -n1 basename | sed 's/.wav//')
```

## 2. Models

The GGUF checkpoints, shared by jinfer and parakeet.cpp:

```bash
bin/jinfer pull mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf \
                mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf \
                mudler/parakeet-cpp-gguf/tdt-0.6b-v3-f16.gguf \
                mudler/parakeet-cpp-gguf/tdt_ctc-110m-q4_k.gguf
```

It prints each path; `jinfer list` prints them again later. sherpa-onnx needs its own export of the
same checkpoint:

```bash
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
tar xf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
```

## 3. Machine

The work scales with physical cores, and SMT siblings cost 35-50%: on the machine below, jinfer
drops from 25.1 to 20.1 RTFx, parakeet.cpp from 11.9 to 7.9, sherpa-onnx from 18.6 to 11.0. jinfer
defaults to the physical core count; the others take what they are given, so every engine is run
with that number.

```bash
lscpu -p=Core | grep -v '^#' | sort -u | wc -l   # physical cores; use as THREADS below
```

A laptop throttles: run on mains power, with nothing else running, and interleave the engines so
each meets the same thermal state. Pin the governor where it is available:

```bash
sudo cpupower frequency-set -g performance
sensors | grep -i tctl                           # watch it between runs
```

## 4. jinfer

```bash
mvn -pl jinfer/jinfer-bench -am -DskipTests package

java --add-modules jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
  -cp jinfer/jinfer-bench/target/jinfer-bench.jar com.qxotic.jinfer.bench.TranscriptionBench \
  --model ~/.cache/huggingface/hub/**/tdt-0.6b-v3-q4_k.gguf \
  --librispeech test-fixtures/librispeech/LibriSpeech/test-clean \
  --limit 100 --dump /tmp/jinfer-q4_k.tsv
```

Those two JVM flags and no others: they are what the native image is built with. The corpus runs on
one reused state, as a server holds one per pipeline, so the timed region is decoding rather than
allocating. `--dump` writes id, audio seconds, decode seconds, hypothesis and reference per
utterance. `-Djinfer.threads=N` overrides the thread count; `--gate <percent>` exits non-zero above
a WER bound, for CI.

The native binary (`make native`, then `bin/jinfer-bench`) measures what the shipped executable
does; the JVM run measures the jars.

## 5. parakeet.cpp

```bash
git clone https://github.com/mudler/parakeet.cpp && cd parakeet.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cd -

parakeet.cpp/build/examples/cli/parakeet-cli bench \
  --model ~/.cache/huggingface/hub/**/tdt-0.6b-v3-q4_k.gguf --decoder tdt --threads 8 \
  --manifest test-fixtures/librispeech/manifest.tsv --json /tmp/cpp-q4_k.json
```

`--decoder tdt` on every checkpoint here, including the hybrid 110m, which jinfer also decodes as
TDT. The JSON carries `files[]` of path, `audio_sec`, `proc_ms` and `text`.

## 6. sherpa-onnx

```bash
pip install sherpa-onnx

python3 - <<'EOF' > /tmp/sherpa.tsv
import pathlib, time, wave, array, sherpa_onnx
model = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{model}/encoder.int8.onnx", decoder=f"{model}/decoder.int8.onnx",
    joiner=f"{model}/joiner.int8.onnx", tokens=f"{model}/tokens.txt",
    model_type="nemo_transducer", num_threads=8)
for row in open("test-fixtures/librispeech/manifest.tsv"):
    path, reference = row.rstrip("\n").split("\t", 1)
    pcm = [s / 32768 for s in array.array("h", wave.open(path).readframes(-1))]
    started = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, pcm)
    recognizer.decode_stream(stream)
    print(f"{pathlib.Path(path).stem}\t{len(pcm) / 16000:.3f}\t{time.time() - started:.3f}"
          f"\t{stream.result.text.strip()}\t{reference}")
EOF
```

Model load sits outside the loop, as in the other two harnesses.

## 7. Scoring

One scorer for every engine, reading both shapes (`*.tsv` dumps and parakeet.cpp's `*.json`). The
run named by `--against` is the baseline for agreement:

```bash
jinfer/scripts/score_asr.py --references test-fixtures/librispeech/manifest.tsv \
  --against /tmp/cpp-q4_k.json /tmp/jinfer-q4_k.tsv /tmp/cpp-q4_k.json /tmp/sherpa.tsv
```

```text
jinfer-q4_k     WER  2.04%   RTFx   25.1   agreement  0.17%
cpp-q4_k        WER  2.04%   RTFx   11.9   agreement  0.00%
sherpa          WER  2.16%   RTFx   18.6   agreement  1.36%
```

Report the median of at least three runs per configuration, and the spread with it.

## 8. Measured

AMD Ryzen 7 PRO 8840U (8 cores, SMT on, laptop, ~75 °C under load), Linux 7.2, GraalVM 25 JVM,
8 threads, median of 3 runs, 2026-09-22. Checkpoints from `mudler/parakeet-cpp-gguf`; sherpa-onnx
runs the int8 ONNX export of the same v3 checkpoint, which is none of these quantizations.

| Model | Engine | WER | RTFx median | Runs min-max | Agreement |
|-------|--------|-----|-------------|--------------|-----------|
| tdt-0.6b-v3 q4_k | jinfer | 2.04% | **25.1** | 24.1-25.1 | 0.17% |
| | parakeet.cpp | 2.04% | 12.6 | 11.9-12.9 | - |
| tdt-0.6b-v3 q8_0 | jinfer | 2.04% | **21.6** | 20.9-22.0 | 0.04% |
| | parakeet.cpp | 2.04% | 14.9 | 13.4-15.2 | - |
| tdt-0.6b-v3 f16 | jinfer | 2.04% | 13.6 | 13.5-15.1 | 0.00% |
| | parakeet.cpp | 2.04% | **15.3** | 14.6-15.7 | - |
| tdt_ctc-110m q4_k | jinfer | 1.99% | **81.2** | 75.2-83.7 | 0.34% |
| | parakeet.cpp | 2.12% | 45.8 | 45.4-46.0 | - |
| v3 int8 ONNX | sherpa-onnx | 2.16% | 18.6 | 18.4-19.7 | 1.27% |

Identical transcripts at f16 and near-identical at q8_0 leave the remaining differences to
quantization rather than to the port. Oracle GraalVM 25.4.4 measures within noise of the GraalVM 25
JVM above.

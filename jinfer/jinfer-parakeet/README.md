<h1 align="center">jinfer-parakeet</h1>

<p align="center"><strong>Automatic Speech Recognition for the JVM</strong></p>

<div align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>

An implementation of [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for
the JVM: blazing fast on ordinary CPUs, competitive with the native implementations. No ONNX Runtime, no whisper.cpp,
no Python runtime. Just the JVM.

</div>

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.mp4"><img src="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.png" alt="Live transcription: final words settle, the provisional tail follows in grey, a waveform tracks the voice" width="820"></a>
</p>
<p align="center"><sub>Live transcription of JFK's 1961 inaugural address. <a href="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.mp4">Play it with sound.</a></sub>

## Highlights

- **Competitive with the reference engines.** Keeps pace
  with [parakeet.cpp](https://github.com/mudler/parakeet.cpp) and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), word for word identical.
- **Matches the reference with 100% accuracy.** At `F16` the transcript matches [parakeet.cpp](https://github.com/mudler/parakeet.cpp) exactly;
  the rest is quantization noise, not drift.
- **Streaming support.** Audio in as it arrives, text out in pieces, with a draft tail that keeps up with the speaker.
- **Word timings and confidence.** Every token carries its audio span and how sure the decoder was.
- **Multilingual.** Parakeet v3 transcribes and punctuates without a language flag.
- **First-class support for GraalVM's Native Image.** Low memory footprint, standalone images, fast startup e.g. 600M model is loaded and ready to transcribe in about 0.2 s.

## Benchmarks

LibriSpeech test-clean, first 100 utterances (901 s), 8 threads on an AMD Ryzen 7 PRO 8840U
laptop, median of 3 runs. The last column scores each run against parakeet.cpp's transcript rather
than against the reference, so 0% means both engines heard exactly the same words.
[How to reproduce this, step by step](WER.md).

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

_RTFx_ is seconds of audio transcribed per wall second, model load excluded. The small model
transcribes an hour of speech in 44 seconds.

For the full matrix - both models, three engines, five quantizations, 1 to 16 threads and three
JVMs, on a second machine - see [BENCHMARKS.md](BENCHMARKS.md).

## Supported models

| Checkpoint | Languages | Parameters | GGUF (`Q8_0`) |
|------------|-----------|------------|---------------|
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | multilingual | 600M | [tdt-0.6b-v3-q8_0.gguf](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-0.6b-v3-q8_0.gguf?download=true) |
| [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | English | 600M | [tdt-0.6b-v2-q8_0.gguf](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-0.6b-v2-q8_0.gguf?download=true) |
| [parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) | English | 1.1B | [tdt-1.1b-q8_0.gguf](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-1.1b-q8_0.gguf?download=true) |
| [parakeet-tdt_ctc-1.1b](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b) | English | 1.1B | [tdt_ctc-1.1b-q8_0.gguf](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-1.1b-q8_0.gguf?download=true) |
| [parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | English | 110M | [tdt_ctc-110m-q8_0.gguf](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q8_0.gguf?download=true) |

Every checkpoint also ships `Q4_K`, `Q5_K`, `Q6_K` and `F16` in
[mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf), named the same way:
swap the quantization in the file name. `Q8_0` balances quality and size, `Q4_K` runs quickest.
The CLI downloads and caches them by reference, so no manual download is needed:

```bash
jinfer pull mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf
```

## Transcribe a file

```bash
jinfer -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf transcribe speech.wav
```

Any format ffmpeg reads, resampled by `jinfer-codecs`. The transcript goes to stdout, so it pipes.

## Live transcription

Raw 16 kHz mono PCM on stdin, transcribed as it arrives. The microphone comes from ffmpeg, so
the capture flags are the only part that differs:

```bash
# Linux (PulseAudio or PipeWire)
ffmpeg -nostats -loglevel error -f pulse -i default -ar 16000 -ac 1 -f s16le - \
  | jinfer -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf transcribe - --raw-pcm
```

```bash
# macOS (AVFoundation; ffmpeg -f avfoundation -list_devices true -i "" names the inputs)
ffmpeg -nostats -loglevel error -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - \
  | jinfer -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf transcribe - --raw-pcm
```

```bash
# Windows (DirectShow; ffmpeg -list_devices true -f dshow -i dummy names the inputs)
ffmpeg -nostats -loglevel error -f dshow -i audio="Microphone" -ar 16000 -ac 1 -f s16le - ^
  | jinfer -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf transcribe - --raw-pcm
```

On a terminal, stderr shows the live view: final words settle into the scrollback, each committed
piece lands in color and fades into the text, the provisional tail follows in grey italics with a
band of light sweeping through it, and a waveform dances with the voice. Doubtful words are
flagged, so a name the model is unsure of stands out. `--theme mint|nord|catppuccin|ember|frost|mono`
picks the palette; colors degrade to 256, to 16, and to plain attributes under `NO_COLOR`.

Redirected, the same run logs final text and partials as plain lines, so it scripts.

## From Java

Through [LangChain4j](../jinfer-langchain4j/README.md), which resolves the model ref and downloads
it on first use:

```java
try (var transcriber = JinferTranscriptionModel.builder()
        .model("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")
        .build()) {
    Transcription transcription = transcriber.transcribe(Path.of("speech.wav"));
    System.out.println(transcription.text());
    for (Transcription.Word word : transcription.words())
        System.out.printf("%s  %.2f  %s%n", word.start(), word.confidence(), word.text());
}
```

Or through [Spring AI](../jinfer-spring-ai/README.md), with `jinfer-spring-ai` on the classpath:

```java
try (var transcriber = JinferTranscriptionModel.builder()
        .model("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")
        .build()) {
    AudioTranscriptionResponse response = transcriber.call(
            new AudioTranscriptionPrompt(new FileSystemResource("speech.wav")));
    System.out.println(response.getResult().getOutput());
}
```

Live audio needs `jinfer-parakeet` directly: feed samples as they arrive, print the final pieces,
and poll the provisional tail for a responsive UI.

```java
static <S extends RuntimeState> void live(TranscriptionModel<?, ?, S> model, float[] pcm) {
    try (S state = model.newState();
            TranscriptionStream stream = model.stream(state)) {
        int chunk = model.sampleRate() / 2; // feed half a second at a time
        for (int at = 0; at < pcm.length; at += chunk) {
            Transcription piece = stream.feed(pcm, at, Math.min(chunk, pcm.length - at));
            System.out.print(piece.text()); // final text, often empty, never revised
            System.err.print("\r" + stream.partial().text()); // provisional, replaced next time
        }
        System.out.println(stream.finish().text()); // the rest, final
    }
}
```

Final pieces never change and join, in order, into the whole transcript. Token times are offsets
from the start of the stream.

## How the streaming works

Audio is decoded in chunks with context on both sides, as NeMo streams Parakeet: 10 s of left
context, a 2 s chunk, 2 s of right context. Only the chunk's tokens become final, while the
decoder carries its state across chunks, so words continue across boundaries. Final text trails the
speaker by 2 to 4 s; the provisional tail, decoded over a shorter window, refreshes twice a second
while speech comes in.

Offline transcription runs the same path with 46 s chunks, so a two hour recording never holds a
two hour attention matrix.

## Model notes

Parakeet v3 emits cased, punctuated text and one timing per token. It consumes mono 16 kHz audio;
anything else is refused rather than silently resampled, since a rate mismatch quietly degrades
recognition. Short windows around digital silence can blank
([NVIDIA-NeMo/Speech#15757](https://github.com/NVIDIA-NeMo/Speech/issues/15757)), so live recording
with a room noise floor transcribes better than padded silence.

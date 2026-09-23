<h1 align="center">jinfer-parakeet</h1>

<p align="center"><strong>Automatic Speech Recognition for the JVM</strong></p>

<div align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>

An implementation of [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for
the JVM: the FastConformer encoder and the TDT transducer decoder, fast on ordinary CPUs, faster
than the reference engines on the same checkpoints. No ONNX Runtime, no whisper.cpp, no PyTorch,
no native library to ship. Just the JVM.

</div>

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.mp4"><img src="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.png" alt="Live transcription: final words settle, the provisional tail follows in grey, a waveform tracks the voice" width="820"></a>
</p>
<p align="center"><sub>Live transcription of JFK's 1961 inaugural address. <a href="https://cdn.jsdelivr.net/gh/qxoticai/assets@e1fe3eacf469cf2962083a924d6daf260759e926/qxotic/parakeet-live-transcription.mp4">Play it with sound.</a></sub>

## Highlights

- **Faster than the reference engines.** On the same checkpoint and machine, 2.0x parakeet.cpp at
  `Q4_K` and 1.35x sherpa-onnx's int8 ONNX export, word for word identical.
- **Same words as the reference.** At `F16` the transcript matches parakeet.cpp exactly; the rest
  is quantization noise, not porting drift.
- **Streaming, not just files.** Audio in as it arrives, final text out in pieces, with a
  provisional tail that keeps up with the speaker.
- **Word timings and confidence.** Every token carries its audio span and how sure the decoder was.
- **Multilingual.** Parakeet v3 transcribes and punctuates without a language flag.
- **Native Image ready.** No runtime to install, and a 600M checkpoint is mapped and ready in
  about 0.2 s.

## Benchmarks

LibriSpeech test-clean, first 100 utterances (901 s), 8 threads on an AMD Ryzen 7 PRO 8840U
laptop, median of 3 runs. Agreement is WER against parakeet.cpp's own transcript, so 0% means the
two engines heard the same words. [How to reproduce this, step by step](WER.md).

| Model | Engine | WER | RTFx | Agreement |
|-------|--------|-----|------|-----------|
| tdt-0.6b-v3 `Q4_K` | **jinfer** | 2.04% | **25.1** | 0.17% |
| | parakeet.cpp | 2.04% | 12.6 | - |
| tdt-0.6b-v3 `Q8_0` | **jinfer** | 2.04% | **21.6** | 0.04% |
| | parakeet.cpp | 2.04% | 14.9 | - |
| tdt-0.6b-v3 `F16` | jinfer | 2.04% | 13.6 | 0.00% |
| | parakeet.cpp | 2.04% | **15.3** | - |
| tdt_ctc-110m `Q4_K` | **jinfer** | 1.99% | **81.2** | 0.34% |
| | parakeet.cpp | 2.12% | 45.8 | - |
| v3 int8 ONNX | sherpa-onnx | 2.16% | 18.6 | 1.27% |

RTFx is seconds of audio transcribed per wall second, model load excluded. The small model
transcribes an hour of speech in 44 seconds.

## Supported checkpoints

TDT checkpoints, including the hybrid TDT-CTC ones, whose TDT head is the one that decodes.
CTC-only and RNN-T checkpoints are refused at load.

| Checkpoint | Languages | Parameters |
|------------|-----------|------------|
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | multilingual | 600M |
| [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | English | 600M |
| [parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) | English | 1.1B |
| [parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | English | 110M |

GGUF conversions of all of them live in
[mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf). Quantizations:
`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, `F32`. `Q8_0` is the balanced choice, `Q4_K` the fastest
here.

## Transcribe a file

```bash
bin/jinfer -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf --transcribe speech.wav
```

Any format ffmpeg reads, resampled by `jinfer-codecs`. The transcript goes to stdout, so it pipes.

## Live transcription

Raw 16 kHz mono PCM on stdin, transcribed as it arrives:

```bash
ffmpeg -nostats -loglevel error -f pulse -i default -ar 16000 -ac 1 -f s16le - \
  | bin/jinfer -m mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q4_k.gguf --transcribe -
```

Capture with `-f pulse -i default` on Linux, `-f avfoundation -i ":0"` on macOS, `-f dshow -i
audio="Microphone"` on Windows.


On a terminal, stderr shows the live view: final words settle into the scrollback, each committed
piece lands in color and fades into the text, the provisional tail follows in grey italics with a
band of light sweeping through it, and a waveform dances with the voice. Doubtful words are
flagged, so a name the model is unsure of stands out. `--theme mint|nord|catppuccin|ember|frost|mono`
picks the palette; colors degrade to 256, to 16, and to plain attributes under `NO_COLOR`.

Redirected, the same run logs final text and partials as plain lines, so it scripts.

## From Java

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-parakeet</artifactId>
  <version>0.2.0</version>
</dependency>
```

A file, with word timings:

```java
try (Arena arena = Arena.ofShared()) {
    TranscriptionModel<?, ?, ?> model =
            Models.loadTranscription(Path.of("tdt-0.6b-v3-q8_0.gguf"), arena);
    Transcription transcription = model.transcribe(AudioCodec.load(Path.of("speech.wav")));
    System.out.println(transcription.text());
    for (Transcription.Word word : transcription.words())
        System.out.printf("%s %.2f %s%n", word.start(), word.confidence(), word.text());
}
```

Live audio: feed samples as they arrive, take the final pieces, poll the provisional tail for a
responsive UI.

```java
static <S extends RuntimeState> void live(TranscriptionModel<?, ?, S> model, float[] chunk) {
    try (S state = model.newState();
            TranscriptionStream stream = model.stream(state)) {
        Transcription piece = stream.feed(chunk); // final text, often empty
        Transcription tail = stream.partial(); // provisional, replaced next time
        System.out.println(piece.text() + " [" + tail.text() + "]");
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

Parakeet v3 emits cased, punctuated text and one timing per token. It expects mono 16 kHz audio;
anything else is refused rather than silently resampled, since a rate mismatch quietly degrades
recognition. Very short windows around digital silence can blank
([NVIDIA-NeMo/Speech#15757](https://github.com/NVIDIA-NeMo/Speech/issues/15757)), so live capture
with a room noise floor transcribes better than padded silence.

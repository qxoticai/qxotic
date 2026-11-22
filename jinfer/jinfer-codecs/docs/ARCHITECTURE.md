# jinfer-codecs — class architecture

Package `com.qxotic.jinfer.codecs` (15 top-level types + `package-info`, ~1600 lines). It turns local media files and
encoded bytes into jinfer's canonical model inputs (`Media.Image` RGB/[0,1]/HWC, `Media.Audio` 16 kHz mono float PCM,
`Media.Video` timestamped frames), encodes synthesized audio back to WAV/PCM16, plays audio locally, and wraps
espeak-ng as a grapheme-to-phoneme tool. Its only Maven dependency is `jinfer-core` (for `Media`, `SpeechSynthesisModel`,
`SpeechOptions`); external tools (ffmpeg, ffprobe, espeak-ng, audio players) are reached as subprocesses. There is **no
`META-INF/services` registration** and no resources: backends are selected by static facades with `-D` overrides and
reflective loading. `package-info` declares the package internal to jinfer (no API or compatibility promise).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a image/audio decoding, 2b video, 2c speech side, 2d subprocess layer
3. [Backend selection](#3-backend-selection)
4. [Decode flows](#4-decode-flows) — 4a ffmpeg image, 4b ffmpeg audio, 4c javasound audio
5. [Video sampling](#5-video-sampling)
6. [Audio encoding and playback](#6-audio-encoding-and-playback)
7. [Espeak grapheme-to-phoneme](#7-espeak-grapheme-to-phoneme)
8. [Subprocess runner](#8-subprocess-runner)
9. [Limits and system properties](#9-limits-and-system-properties)

| Class | Role |
|---|---|
| `ImageCodec` | Static facade: lazily selects and caches an `ImageDecoder` (`ffmpeg` / `imageio`); `load`, `decode`, `decoder()`; holds `MAX_PIXELS` and `checkDimensions` |
| `ImageDecoder` | Interface: `load(Path)`, `decode(byte[])` → `Media.Image`; `name()` |
| `FfmpegImageDecoder` | ffmpeg-based image decoder; forces the demuxer from magic bytes; parses binary PPM (P6), also reused for video frames |
| `ImageIoDecoder` | `javax.imageio` decoder (PNG/JPEG/GIF/BMP/TIFF), JVM only, loaded reflectively |
| `AudioCodec` | Static facade: selects and caches an `AudioDecoder` (`ffmpeg` / `javasound`); also hand-written encoders `wav(Audio)` and `pcm16(Audio)` |
| `AudioDecoder` | Interface: `load(Path)`, `decode(byte[])` → 16 kHz mono `Media.Audio`; `name()` |
| `FfmpegAudioDecoder` | ffmpeg decode+resample+downmix to f32le in one pass; owns `SAMPLE_RATE` and the max-minutes limit helpers |
| `JavaSoundAudioDecoder` | `javax.sound.sampled` decoder (WAV/AIFF/AU) via the JDK format converter, JVM only, loaded reflectively |
| `VideoCodec` | Interface: primitives `totalDuration`, `framePeriod`, `framesAt` + default sampling policies `frameAt`, `uniform`, `every`, `span`; `ffmpeg()` factory |
| `FfmpegVideoCodec` | Singleton `VideoCodec`: one cached ffprobe run per file, all frames extracted in one ffmpeg process |
| `VideoSampler` | `@FunctionalInterface` `Path → Media.Video` policy; constant `UNIFORM` |
| `AudioPlayer` | `AutoCloseable` local playback via afplay / aplay / ffplay / PowerShell SoundPlayer; `play`, `stream`; nested `Failed` exception |
| `Espeak` | espeak-ng / espeak wrapper: `find()`, `ipa`, `tiedIpa`, `TIE` |
| `Codecs` | Package-private selection helpers: `nativeImage()`, `choice()`, `ffmpegOnPath()`, `reflect()` |
| `Subprocess` | Package-private external-tool runner with timeout, output caps, stderr capture; nested `OutputLimitExceeded` |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph callers["Callers (depend on jinfer-codecs)"]
        Server["jinfer-server<br/>MediaParts, TranscriptionServer,<br/>RetrievalServer, DecisionServer, Generation, Server"]
        LC4J["jinfer-langchain4j<br/>Mappings, Jinfer*Model"]
        Spring["jinfer-spring-ai"]
        Cli["jinfer-cli Speak / Transcribe<br/>jinfer-tts TtsCli"]
        Chat["jinfer-chat Models<br/>decoder name in model identity"]
        TTS["jinfer-kokoro KokoroTTS, Misaki<br/>jinfer-inflect2 InflectTTS"]
    end
    subgraph codecs["jinfer-codecs"]
        IC["ImageCodec"]
        AC["AudioCodec"]
        VC["VideoCodec / VideoSampler"]
        AP["AudioPlayer"]
        ES["Espeak"]
        SP["Subprocess"]
    end
    subgraph core["jinfer-core"]
        Media["Media: Image | Audio | Video"]
        SSM["SpeechSynthesisModel, SpeechOptions"]
        Ph["Phonemizer"]
    end
    Tools["external tools on PATH:<br/>ffmpeg, ffprobe, espeak-ng, aplay, afplay, ffplay, powershell"]

    Server --> IC
    Server --> AC
    Server --> VC
    LC4J --> IC
    LC4J --> AC
    LC4J --> VC
    Spring --> codecs
    Cli --> AP
    Cli --> AC
    Chat --> IC
    Chat --> AC
    TTS --> ES
    TTS -.->|"builds Phonemizer.ipa over Espeak"| Ph
    IC --> Media
    AC --> Media
    VC --> Media
    AP --> SSM
    IC --> SP
    AC --> SP
    VC --> SP
    ES --> SP
    SP --> Tools
    AP --> Tools
```

Decoding stops at `Media`: a model's `MediaProjector` (jinfer-core) turns decoded media into embedding rows; codecs never
see a model. No codec fetches remote content.

---

## 2. Class diagrams

### 2a. Image and audio decoding

```mermaid
classDiagram
    direction LR
    class Media { <<sealed interface, external>> }
    class Image { <<record, external>> }
    class Audio { <<record, external>> }
    Media <|.. Image
    Media <|.. Audio

    class ImageDecoder {
        <<interface>>
        load(Path) Image
        decode(byte[]) Image
        name() String
    }
    class AudioDecoder {
        <<interface>>
        load(Path) Audio
        decode(byte[]) Audio
        name() String
    }
    class ImageCodec {
        <<final, static facade>>
        MAX_PIXELS = 4096x4096
        -decoder volatile ImageDecoder
        load(Path) Image
        decode(byte[]) Image
        decoder() ImageDecoder
        select() ImageDecoder
        checkDimensions(w, h)
    }
    class AudioCodec {
        <<final, static facade>>
        -decoder volatile AudioDecoder
        load(Path) Audio
        decode(byte[]) Audio
        decoder() AudioDecoder
        select() AudioDecoder
        wav(Audio) byte[]
        pcm16(Audio) byte[]
    }
    class FfmpegImageDecoder {
        <<final>>
        demuxer(head) String
        parsePpm(byte[]) Image
        parsePpm(byte[], int[] pos) Image
    }
    class ImageIoDecoder { <<final, JVM only>> }
    class FfmpegAudioDecoder {
        <<final>>
        SAMPLE_RATE = 16000
        maxMinutes() int
        maxSamples(int) int
        tooLong(int, IOException) IOException
    }
    class JavaSoundAudioDecoder {
        <<final, JVM only>>
        readBounded(InputStream, int) byte[]
    }
    class Codecs {
        <<final, package-private>>
        nativeImage() boolean
        choice(property, jdkBackend) String
        ffmpegOnPath() boolean
        reflect~T~(className, property) T
    }

    ImageDecoder <|.. FfmpegImageDecoder
    ImageDecoder <|.. ImageIoDecoder
    AudioDecoder <|.. FfmpegAudioDecoder
    AudioDecoder <|.. JavaSoundAudioDecoder
    ImageCodec --> ImageDecoder : caches
    AudioCodec --> AudioDecoder : caches
    ImageCodec ..> Codecs : choice / reflect
    AudioCodec ..> Codecs : choice / reflect
    ImageCodec ..> FfmpegImageDecoder : new
    AudioCodec ..> FfmpegAudioDecoder : new
    Codecs ..> ImageIoDecoder : Class.forName
    Codecs ..> JavaSoundAudioDecoder : Class.forName
    FfmpegImageDecoder ..> ImageCodec : checkDimensions
    ImageIoDecoder ..> ImageCodec : checkDimensions
    JavaSoundAudioDecoder ..> FfmpegAudioDecoder : SAMPLE_RATE, maxMinutes, tooLong
    FfmpegImageDecoder ..> Image : creates
    ImageIoDecoder ..> Image : creates
    FfmpegAudioDecoder ..> Audio : creates
    JavaSoundAudioDecoder ..> Audio : creates
```

The two JDK-backed decoders are never referenced statically: `Codecs.reflect` receives their class name as a
non-constant argument, so native-image cannot fold the `Class.forName` and drag `java.desktop` /
`javax.sound.sampled` into the image.

### 2b. Video

```mermaid
classDiagram
    direction LR
    class Video { <<record, external>> frames List~Frame~ }
    class Frame { <<record, external>> image Image; timestamp Duration }
    Video *-- Frame

    class VideoCodec {
        <<interface>>
        DEFAULT_NUM_FRAMES = 32
        ffmpeg()$ VideoCodec
        name() String
        totalDuration(Path) Duration
        framePeriod(Path) Duration
        framesAt(Path, Duration...) Video
        frameAt(Path, Duration) Frame
        uniform(Path) Video
        uniform(Path, int) Video
        uniform(Path, from, to, int) Video
        every(Path, step, maxFrames) Video
        span(Path, int) Video
        span(Path, from, to, int) Video
    }
    class FfmpegVideoCodec {
        <<final, singleton>>
        INSTANCE$
        -last volatile Probe
        seekTime(Duration) String
    }
    class Key { <<record, private>> video; size; modified }
    class Probe { <<record, private>> key; duration; framePeriod }
    class VideoSampler {
        <<functional interface>>
        UNIFORM$
        sample(Path) Video
    }
    class FfmpegImageDecoder { <<final>> }

    VideoCodec <|.. FfmpegVideoCodec
    VideoCodec ..> FfmpegVideoCodec : ffmpeg returns INSTANCE
    FfmpegVideoCodec *-- Probe : last
    Probe *-- Key
    FfmpegVideoCodec ..> FfmpegImageDecoder : parsePpm per frame
    FfmpegVideoCodec ..> Video : creates
    VideoSampler ..> VideoCodec : UNIFORM uses ffmpeg uniform
```

`VideoCodec` is a *sampler*, not a decoder: there is no `byte[]` overload and no whole-video decode. The policies
(`uniform`, `every`, `span`) are default methods of pure timestamp arithmetic over the three primitives.

### 2c. Speech side: encoding, playback, phonemization

```mermaid
classDiagram
    direction LR
    class SpeechSynthesisModel~C,W,S~ {
        <<interface, external>>
        sampleRate() int
        speak(text, options, sink)
    }
    class SpeechOptions { <<interface, external>> }
    class Phonemizer { <<interface, external>> }
    class Audio { <<record, external>> }

    class AudioPlayer {
        <<final, AutoCloseable>>
        -process Process
        -command String[]
        -wav Path
        -firstClip boolean
        play(Audio)$
        stream(model, text, options, firstAudio)$
        streamsWav(osName)$ boolean
        fileCommands(osName, wav)$ String[][]
        close()
    }
    class Failed {
        <<AudioPlayer.Failed>>
        status() int
    }
    class IOException { <<external>> }
    class AudioCodec { <<final>> wav(Audio); pcm16(Audio) }
    class Espeak {
        <<final>>
        TIE = "^"
        -binary String
        -timeout Duration
        find()$ Optional~Espeak~
        ipa(run, language) String
        tiedIpa(run, language) String
    }

    IOException <|-- Failed
    AudioPlayer *-- Failed
    AudioPlayer ..> AudioCodec : wav / pcm16
    AudioPlayer ..> SpeechSynthesisModel : speak with clip sink
    AudioPlayer ..> SpeechOptions
    AudioPlayer ..> Audio
    Espeak ..> Phonemizer : speech ports wrap it via Phonemizer.ipa
```

### 2d. Subprocess layer

```mermaid
classDiagram
    direction LR
    class Subprocess {
        <<final, package-private>>
        TIMEOUT = 2 min
        MAX_OUTPUT_BYTES = 256 MiB
        MAX_ERROR_BYTES = 64 KiB
        run(cmd, stdin)$ byte[]
        run(cmd, stdin, timeout, maxOutputBytes)$ byte[]
    }
    class OutputLimitExceeded { <<Subprocess.OutputLimitExceeded>> }
    class IOException { <<external>> }
    IOException <|-- OutputLimitExceeded
    Subprocess *-- OutputLimitExceeded

    Codecs ..> Subprocess : ffmpeg -version
    FfmpegImageDecoder ..> Subprocess
    FfmpegAudioDecoder ..> Subprocess
    FfmpegVideoCodec ..> Subprocess : ffprobe + ffmpeg
    Espeak ..> Subprocess
    JavaSoundAudioDecoder ..> OutputLimitExceeded : readBounded throws
    FfmpegAudioDecoder ..> OutputLimitExceeded : catches, tooLong
```

`AudioPlayer` does **not** use `Subprocess`: it starts its own long-lived `ProcessBuilder` processes (stdout discarded,
stderr inherited) because it streams to the player's stdin and waits on exit in `close()`.

---

## 3. Backend selection

`ImageCodec.decoder()` and `AudioCodec.decoder()` use double-checked locking on a `volatile` field; the first call runs
`select()`, which asks `Codecs.choice(property, jdkBackend)`.

```mermaid
flowchart TD
    A["decoder() first call"] --> B{"-D property set?<br/>jinfer.imageDecoder / jinfer.audioDecoder"}
    B -->|yes| L["lowercase value"]
    B -->|no| C{"Codecs.nativeImage()?<br/>org.graalvm.nativeimage.imagecode, read at run time"}
    C -->|yes| F["ffmpeg"]
    C -->|no| D{"Codecs.ffmpegOnPath()?<br/>ffmpeg -version, 2 s timeout"}
    D -->|yes| F
    D -->|no| J["JDK backend: imageio / javasound"]
    L --> S{"switch value"}
    F --> S
    J --> S
    S -->|ffmpeg| NF["new FfmpegImageDecoder / FfmpegAudioDecoder"]
    S -->|"imageio / javasound"| R["Codecs.reflect(className, property)"]
    S -->|other| X["IllegalArgumentException"]
    R -->|"ReflectiveOperationException or LinkageError"| E["IllegalStateException naming -Dproperty=ffmpeg"]
```

A selected backend is the only one asked: there is no runtime fallback from one decoder to the other, and `reflect`
refuses rather than silently swapping. `nativeImage()` is deliberately not a `static final` so it is not
constant-folded at image build time. `VideoCodec` has a single backend (`VideoCodec.ffmpeg()`), with no selection.

---

## 4. Decode flows

### 4a. ffmpeg image decode

```mermaid
sequenceDiagram
    participant C as Caller
    participant IC as ImageCodec
    participant FD as FfmpegImageDecoder
    participant SP as Subprocess
    participant FF as ffmpeg process
    C->>IC: decode(bytes)
    IC->>IC: decoder()
    IC->>FD: decode(bytes)
    FD->>FD: demuxer(bytes) - magic bytes to png_pipe / jpeg_pipe / gif / webp_pipe / bmp_pipe / tiff_pipe / null
    FD->>SP: run(ffmpeg [-f demuxer] -i pipe:0 -f image2pipe -vcodec ppm -pix_fmt rgb24 -, bytes)
    SP->>FF: start, stdin on daemon thread
    FF-->>SP: P6 PPM on stdout
    SP-->>FD: byte[]
    FD->>FD: parsePpm - header tokens, maxval 255, ImageCodec.checkDimensions
    FD-->>C: Media.Image(values in 0..1, h, w, 3)
```

`load(Path)` reads the first 16 bytes of the file for the demuxer and passes the path as `-i`. The forced demuxer
prevents ffmpeg's probe from misidentifying a broken file (comment cites a corrupt PNG probed as CD+G graphics).
`ImageIoDecoder` instead picks the first `ImageIO` reader, checks dimensions from the reader before decoding, and
converts `BufferedImage.getRGB` to the same HWC float layout.

### 4b. ffmpeg audio decode

```mermaid
sequenceDiagram
    participant C as Caller
    participant AC as AudioCodec
    participant FA as FfmpegAudioDecoder
    participant SP as Subprocess
    C->>AC: load(path) / decode(bytes)
    AC->>FA: load / decode
    FA->>FA: maxMinutes() from -Djinfer.codecs.maxAudioMinutes, default 60
    FA->>SP: run(ffmpeg -i input -ar 16000 -ac 1 -f f32le -, data, 2 min, maxSamples*4)
    alt output exceeds cap
        SP-->>FA: OutputLimitExceeded
        FA-->>C: IOException "audio longer than N minutes"
    else ok
        SP-->>FA: raw float32 LE
        FA->>FA: toAudio - clamp to -1..1, non-finite to 0
        FA-->>C: Media.Audio(pcm, 16000, 1)
    end
```

### 4c. javax.sound audio decode

```mermaid
flowchart LR
    I["AudioSystem.getAudioInputStream<br/>file or BufferedInputStream"] --> T["target AudioFormat<br/>PCM_SIGNED 16 kHz 16-bit mono LE"]
    T --> V["AudioSystem.getAudioInputStream target, in<br/>JDK converter resamples + downmixes"]
    V --> R["readBounded maxSamples*2 bytes"]
    R -->|over cap| O["OutputLimitExceeded -> FfmpegAudioDecoder.tooLong"]
    R --> P["short / 32768f -> Media.Audio 16000 Hz mono"]
    I -->|"UnsupportedAudioFileException or IllegalArgumentException"| U["IOException: reads WAV, AIFF, AU; use ffmpeg"]
```

---

## 5. Video sampling

```mermaid
sequenceDiagram
    participant C as Caller e.g. MediaParts / RetrievalServer
    participant S as VideoSampler
    participant V as FfmpegVideoCodec
    participant SP as Subprocess
    C->>S: sample(path)
    S->>V: uniform(path, n) / every(path, step, max) / span(...)
    V->>V: totalDuration(path), framePeriod(path)
    V->>V: probe(path) - Key(path, size, mtime) vs cached last
    opt cache miss
        V->>SP: ffprobe -select_streams v:0 -show_entries format=duration:stream=r_frame_rate -of csv=p=0
        SP-->>V: two lines - r_frame_rate, duration
        V->>V: store Probe in last
    end
    V->>V: compute timestamps, clamp to duration - framePeriod
    V->>V: framesAt(path, timestamps)
    V->>SP: one ffmpeg: per timestamp -ss t -i path, filter_complex trim end_frame=1 + concat + setpts=N, -fps_mode passthrough, image2pipe ppm rgb24
    SP-->>V: concatenated P6 PPM stream
    loop k in 0..n-1
        V->>V: FfmpegImageDecoder.parsePpm(bytes, pos) - throws if fewer than n frames
    end
    V-->>C: Media.Video(List of Frame(image, timestamp))
```

Policy formulas (from the default methods):

| Method | Timestamps |
|---|---|
| `uniform(v, from, to, n)` | centered: `from + k*(window/n) + window/(2n)`, `to` clamped to duration, each point clamped to the last frame |
| `uniform(v)` | `uniform(v, DEFAULT_NUM_FRAMES = 32)` over the whole source |
| `every(v, step, maxFrames)` | `duration/step` points (≥1), thinned like `numpy.linspace(0, count-1, kept).astype(int)`, last kept exact |
| `span(v, from, to, n)` | fencepost inclusive, `n ≥ 2`, `to` clamped to `duration - framePeriod` |
| `frameAt(v, t)` | `framesAt(v, t).frames().get(0)` |

Callers observed: `VideoSampler.UNIFORM` (OpenAI chat in jinfer-server, langchain4j default); `every(v, 1 s, 32)` for
embeddings (`RetrievalServer`, langchain4j `JinferEmbeddingModel`).

---

## 6. Audio encoding and playback

`AudioCodec.wav` writes a 44-byte RIFF/WAVE PCM header plus `pcm16` samples; `pcm16` rejects non-mono audio and clamps
`sample * Short.MAX_VALUE` to 16-bit LE. The encode side never touches the decoder backend (so a native binary does not
start shelling out to ffmpeg to encode).

```mermaid
sequenceDiagram
    participant C as Caller e.g. TtsCli / Speak
    participant P as AudioPlayer
    participant M as SpeechSynthesisModel
    participant OS as player process
    C->>P: stream(model, text, options, firstAudio)
    P->>M: sampleRate()
    alt Linux / other: raw PCM pipe
        P->>OS: start aplay -f S16_LE ... - , else ffplay -f s16le ...
        P->>M: speak(text, options, clip sink)
        loop each clip
            M-->>P: clip
            P->>OS: offer(AudioCodec.pcm16(clip)) to stdin, false stops synthesis
        end
    else macOS / Windows: WAV per clip
        P->>M: speak(text, options, clip sink)
        loop each clip
            M-->>P: clip
            P->>P: playClip - close previous, write temp WAV
            P->>OS: afplay file / powershell SoundPlayer, else ffplay
        end
    end
    P->>P: firstAudio.run() after first clip
    P->>P: close() - close stdin, waitFor, delete temp WAV
    P-->>C: throws Failed(status) on nonzero exit
```

`start(candidates...)` tries each command in order; only a launch failure moves on to the next candidate. A clip whose
sample rate differs from `model.sampleRate()` raises `IllegalArgumentException`. `play(audio)` is the one-shot WAV path.

---

## 7. Espeak grapheme-to-phoneme

```mermaid
flowchart LR
    F["Espeak.find()"] -->|"espeak-ng --version, then espeak --version<br/>2 s probe each"| O["Optional of Espeak(binary, 5 s)"]
    O --> T["tiedIpa(run, language)"]
    T --> N{"run contains NUL?"}
    N -->|yes| X["IllegalArgumentException"]
    N -->|no| R["Subprocess.run: binary --ipa -q -v lang --tie=^ --stdin<br/>stdin = run + newline, 1 MiB cap"]
    R --> C["newlines to spaces, strip '_',<br/>drop language markers like (en), collapse spaces"]
    C --> I["ipa = tiedIpa minus TIE"]
    R -->|IOException| U["UncheckedIOException"]
```

Consumers: `KokoroTTS` requires espeak (`tiedIpa` + `Misaki` dialect rewrite, which maps `Espeak.TIE`), and
`InflectTTS` uses it optionally as the lexicon fallback (`ipa(run, "en-us")`); both wrap it in `Phonemizer.ipa`
(jinfer-core). The text goes on stdin, never as an argument, and the trailing newline is required for espeak to
phonemize the last word completely.

---

## 8. Subprocess runner

```mermaid
flowchart TD
    S["Subprocess.run(cmd, stdin, timeout, maxOutputBytes)"] --> V["validate timeout and cap"]
    V --> P["ProcessBuilder.start<br/>failure: 'tool could not run: install it and put it on PATH'"]
    P --> I{"stdin?"}
    I -->|yes| D1["daemon tool-stdin writes bytes"]
    I -->|no| D0["close stdin"]
    D1 --> O
    D0 --> O
    O["daemon tool-stdout: readLimited to cap<br/>daemon tool-stderr: drainCapped 64 KiB"] --> W{"waitFor timeout"}
    W -->|timeout| K["destroyForcibly, join, IOException timed out"]
    W -->|exited| J["join drains"]
    J --> F{"stdout failure?"}
    F -->|"yes, e.g. OutputLimitExceeded"| T["rethrow"]
    F -->|no| E{"exit code 0?"}
    E -->|no| Z["IOException: tool exited N + stderr"]
    E -->|yes| R["return stdout bytes"]
```

Defaults for the two-argument overload: 2 minutes, 256 MiB. Callers override: `ffmpeg -version` probe 2 s/64 KiB,
ffprobe 30 s/64 KiB, audio 2 min/`maxSamples*4`, espeak 5 s/1 MiB.

---

## 9. Limits and system properties

| Knob | Where | Effect |
|---|---|---|
| `-Djinfer.imageDecoder=ffmpeg\|imageio` | `ImageCodec.select` | force image backend |
| `-Djinfer.audioDecoder=ffmpeg\|javasound` | `AudioCodec.select` | force audio backend |
| `-Djinfer.codecs.maxAudioMinutes` | `FfmpegAudioDecoder.maxMinutes` | default 60; validated per decode in `[1, MAX_MINUTES_CEILING]`; shared by `JavaSoundAudioDecoder` |
| `ImageCodec.MAX_PIXELS` | `checkDimensions` | 4096 × 4096 pixels; enforced by both image decoders and per video frame |
| `org.graalvm.nativeimage.imagecode` | `Codecs.nativeImage` | read at run time; native images always use ffmpeg |
| `os.name` | `AudioPlayer` | Mac/Windows → WAV per clip; otherwise raw PCM pipe to aplay/ffplay |

`ImageCodec.decoder().name()` and `AudioCodec.decoder().name()` are part of a loaded model's identity string in
jinfer-chat `Models`, because two backends may decode the same bytes differently (media cache keys).

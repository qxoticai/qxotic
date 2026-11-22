# jinfer-tts — class architecture

Package `com.qxotic.jinfer.tts` (3 classes, ~450 lines). It is the generic **text-to-speech command line and
native executable** (`jinfer-tts`, main class `TtsCli`) for every speech model that `jinfer-models-all` puts on the
classpath (today `jinfer-kokoro` and `jinfer-inflect2`). It contains no model code: it parses options, locates the
model and its companions (plain files or `z://` entries of a ZIP overlay appended to the executable), loads the model
through `Models.loadSpeech` and writes, plays or streams the audio. The module is not published to Maven Central.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a module classes, 2b external contracts
3. [Command-line parsing](#3-command-line-parsing-options)
4. [Main flow](#4-main-flow-ttscli)
5. [Self-archive](#5-self-archive-selfarchive) — 5a entry resolution, 5b reading an entry
6. [Loading sequences](#6-loading-sequences) — 6a file model, 6b archived model
7. [Output modes](#7-output-modes)
8. [Build, packaging and resources](#8-build-packaging-and-resources)

| Class | Role |
|---|---|
| `TtsCli` | `public final` entry point: `main` maps failures to exit codes; `run` decides whether an archive is needed, lists entries or loads the model, resolves companions, owns the `Arena` and temp files; `use` / `write` drive synthesis |
| `Options` | package-private `record`: the parsed command line (`model`, `companions`, `text`, `output`, `speed`, `play`, `stream`, `list`, `help`, `archive`); `parse(String[])`, `usage(PrintStream)` |
| `SelfArchive` | package-private `AutoCloseable`: a ZIP (commons-compress `ZipFile`) read from the executable itself or a named file; resolves symlinks, demands `STORED` entries, exposes bounded positional channels and temp-file extraction |
| `SelfArchive.Entry` | nested `record (name, offset, size)`: the data range of one usable entry inside the archive file |

There is **no SPI registration** in this module (no `META-INF/services`): it is a consumer of the
`ModelProvider` SPI, which `Models` discovers through `ServiceLoader` in `jinfer-chat`.

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    User["user / shell"]
    subgraph tts["jinfer-tts"]
        CLI["TtsCli"]
        OPT["Options"]
        SA["SelfArchive"]
    end
    subgraph chat["jinfer-chat"]
        Models["Models<br/>loadSpeech, ServiceLoader"]
        MP["ModelProvider SPI"]
    end
    subgraph core["jinfer-core"]
        SSM["SpeechSynthesisModel"]
        SO["SpeechOptions"]
        AR["Arenas"]
        MA["Media.Audio"]
    end
    subgraph codecs["jinfer-codecs"]
        AC["AudioCodec"]
        AP["AudioPlayer"]
    end
    subgraph models["jinfer-models-all, runtime"]
        KP["KokoroProvider"]
        IP["Inflect2Provider"]
    end
    GG["gguf: GGUF"]
    CC["commons-compress: ZipFile"]
    Players["afplay / PowerShell SoundPlayer / aplay / ffplay"]

    User --> CLI
    CLI --> OPT
    CLI --> SA --> CC
    CLI --> GG
    CLI --> Models --> MP
    KP -.->|implements| MP
    IP -.->|implements| MP
    MP -->|produces| SSM
    CLI --> SSM
    CLI --> AR
    CLI --> AC
    CLI --> AP --> Players
    SSM --> MA
```

Dependencies from `pom.xml`: compile `jinfer-core`, `jinfer-codecs`, `jinfer-chat`, `gguf`, `commons-compress`;
runtime `jinfer-models-all` (all model providers), `jam-native`, `jam-vector` (kernel back-ends).

---

## 2. Class diagrams

### 2a. Module classes

```mermaid
classDiagram
    direction LR
    class TtsCli {
        <<final>>
        -String SELF_ARCHIVE = "z://"
        -String DEFAULT_ENTRY = "default.gguf"
        +main(String[] args)$
        -run(Options options)$
        -resolveCompanions(Map sources, SelfArchive archive, List extracted)$ Map
        -archived(String source)$ boolean
        -entryName(String source)$ String
        -use(SpeechSynthesisModel model, Options options)$
        -speechOptions(Options options)$ SpeechOptions
        -write(SpeechSynthesisModel model, Options options)$
        -printFirstAudio(long start)$
    }
    class Options {
        <<record>>
        String model
        Map companions
        String text
        Path output
        Double speed
        boolean play
        boolean stream
        boolean list
        boolean help
        Path archive
        +parse(String[] args)$ Options
        +usage(PrintStream to)$
        -value(String[] args, int index, String flag)$ String
    }
    class SelfArchive {
        <<final>>
        -int MAX_SYMLINK_DEPTH = 8
        -Path path
        -ZipFile zip
        -FileChannel channel
        +open()$ SelfArchive
        +open(Path file)$ SelfArchive
        +path() Path
        +fileChannel() FileChannel
        +entry(String name) Entry
        -resolve(String name, String as, Set seen) Entry
        +entries() List~Entry~
        +channel(Entry entry) ReadableByteChannel
        +extract(Entry entry) Path
        +close()
    }
    class Entry {
        <<record>>
        String name
        long offset
        long size
    }
    class AutoCloseable { <<interface, external>> }
    class ZipFile { <<external>> }
    class FileChannel { <<external>> }
    class ReadableByteChannel { <<interface, external>> }

    TtsCli ..> Options : parse / usage
    TtsCli ..> SelfArchive : open
    SelfArchive ..|> AutoCloseable
    SelfArchive *-- ZipFile
    SelfArchive *-- FileChannel
    SelfArchive ..> Entry : nested record, creates
    SelfArchive ..> ReadableByteChannel : anonymous bounded view
```

### 2b. External contracts used by `TtsCli`

```mermaid
classDiagram
    direction LR
    class TtsCli { <<final>> }
    class Models {
        <<external>>
        +loadSpeech(Path path, Arena arena, Map companions)$ SpeechSynthesisModel
        +loadSpeech(FileChannel ch, GGUF gguf, long baseOffset, long byteSize, Path path, Arena arena, Map companions)$ SpeechSynthesisModel
    }
    class ModelProvider {
        <<interface, external>>
        +loadSpeech(FileChannel ch, GGUF gguf, Path path, Arena arena, Map companions) SpeechSynthesisModel
        +companionFiles() Map
    }
    class SpeechSynthesisModel~C,W,S~ {
        <<interface, external>>
        +sampleRate() int
        +speak(String text, SpeechOptions options) Media.Audio
        +speak(String text, SpeechOptions options, Predicate sink)
    }
    class SpeechOptions {
        <<interface, external>>
        +NONE$
        +speed(double speed)$ SpeechOptions
        +speed() Double
    }
    class Audio {
        <<record, external>>
        float[] pcm
        int sampleRate
        int channels
    }
    class Arenas {
        <<external>>
        +newCrossThread()$ Arena
        +close(Arena arena)$
    }
    class GGUF {
        <<interface, external>>
        +read(ReadableByteChannel ch)$ GGUF
        +at(long base) GGUF
    }
    class AudioCodec {
        <<external>>
        +wav(Media.Audio audio)$ byte[]
    }
    class AudioPlayer {
        <<external>>
        +play(Media.Audio audio)$
        +stream(SpeechSynthesisModel model, String text, SpeechOptions options, Runnable firstAudio)$
    }
    class Failed {
        <<external>>
        +status() int
    }
    class KokoroProvider { <<external>> }
    class Inflect2Provider { <<external>> }

    TtsCli ..> Models : loadSpeech
    TtsCli ..> GGUF : read header
    TtsCli ..> Arenas
    TtsCli --> SpeechSynthesisModel
    TtsCli ..> SpeechOptions
    TtsCli ..> AudioCodec
    TtsCli ..> AudioPlayer
    TtsCli ..> Failed : catches, exit status
    Models ..> ModelProvider : ServiceLoader select by architecture
    KokoroProvider ..|> ModelProvider
    Inflect2Provider ..|> ModelProvider
    ModelProvider ..> SpeechSynthesisModel : creates
    SpeechSynthesisModel ..> Audio : produces
    AudioPlayer ..> Failed : nested class, throws
```

`AudioPlayer.Failed` extends `IOException`. Provider companion names (from `companionFiles()`):
`KokoroProvider` → `voice`, `Inflect2Provider` → `lexicon`. `Models` rejects companions the selected provider does
not accept (`requireAccepted`).

---

## 3. Command-line parsing (`Options`)

```mermaid
flowchart TD
    A["Options.parse args"] --> B{"args empty?"}
    B -->|yes| H["help = true"]
    B -->|no| L["for each arg"]
    H --> L
    L --> S{"switch arg"}
    S -->|"--model X"| M["model = X"]
    S -->|"--with NAME=SRC"| W{"'=' present, name and source non-empty?"}
    W -->|no| E1["IllegalArgumentException"]
    W -->|yes| W2{"name already given?"}
    W2 -->|yes| E1
    W2 -->|no| W3["companions.put NAME, SRC"]
    S -->|"--text / --output / --archive"| V["store value"]
    S -->|"--speed R"| SP{"number, finite, > 0?"}
    SP -->|no| E1
    SP -->|yes| SP2["speed = R"]
    S -->|"--play / --stream / --list / -h --help"| F["set flag"]
    S -->|"other starting with -"| E1
    S -->|"positional"| P{"model already set?"}
    P -->|yes| E1
    P -->|no| M
    M & W3 & V & SP2 & F --> L
    L -->|done| C{"play and stream?"}
    C -->|yes| E1
    C -->|no| R["new Options, Map.copyOf companions"]
```

Defaults: `text = "Hello world."`, `output = output.wav`, `speed = null` (model default), all flags false.
A missing flag value raises `IllegalArgumentException("<flag> needs a value")`.

---

## 4. Main flow (`TtsCli`)

```mermaid
flowchart TD
    main["main args"] --> parse["Options.parse"]
    parse --> run["run options"]
    run --> help{"help?"}
    help -->|yes| usage["Options.usage System.out"]
    help -->|no| need{"list, or no model, or model / any companion starts with z://?"}
    need -->|no| noarch["archive = null"]
    need -->|yes| which{"--archive given?"}
    which -->|no| self["SelfArchive.open: current process executable"]
    which -->|yes| named["SelfArchive.open path"]
    self & named --> list{"list?"}
    noarch --> load
    list -->|yes| print["print entries name + MB, close archive"]
    list -->|no| load["arena = Arenas.newCrossThread"]
    load --> comp["resolveCompanions: z:// entries extracted to temp files"]
    comp --> src["source = model, or z://default.gguf"]
    src --> arch{"source archived?"}
    arch -->|yes| la["load from archive entry, see 6b"]
    arch -->|no| lf["Models.loadSpeech Path, arena, companions"]
    la & lf --> use["use model, options"]
    use --> fin["finally: Arenas.close arena; delete extracted temp files; close archive"]
```

Error mapping in `main`:

| Exception | Message | Exit |
|---|---|---|
| `IllegalArgumentException`, `UnsupportedOperationException` | `jinfer-tts: <msg>` + usage on stderr | 2 |
| `AudioPlayer.Failed` | `jinfer-tts: <msg>` | `playerQuit.status()` |
| `UncheckedIOException` | cause's message | 1 |
| `IOException` | message | 1 |

Because `Failed` extends `IOException`, it is caught first. Without a model argument the CLI reads
`z://default.gguf`, so a packaged executable speaks with no arguments other than options (note: with *no* arguments
at all, `help` is true and usage is printed).

---

## 5. Self-archive (`SelfArchive`)

The executable may carry a ZIP appended at its end. `SelfArchive.open()` finds the executable through
`ProcessHandle.current().info().command()`; `open(Path)` reads any file (used for `--archive`, e.g. under
`java -jar`). Both a commons-compress `ZipFile` (directory and metadata) and a raw read-only `FileChannel`
(data access) are opened on the same file; `close()` closes both via try-with-resources.

### 5a. Entry resolution (`entry` → `resolve`)

```mermaid
flowchart TD
    E["entry name"] --> R["resolve name, as = name, seen = empty"]
    R --> C1{"name already in seen?"}
    C1 -->|yes| X1["IOException: symlink cycle"]
    C1 -->|no| C2{"seen size > 8?"}
    C2 -->|yes| X2["IOException: symlinks too deep"]
    C2 -->|no| G["zip.getEntry name"]
    G --> C3{"exists?"}
    C3 -->|no| X3["IOException: entry not found"]
    C3 -->|yes| L{"isUnixSymlink?"}
    L -->|yes| T["read target text; absolute: strip root; relative: resolveSibling + normalize"]
    T --> C4{"escapes with '..' or empty absolute?"}
    C4 -->|yes| X4["IOException"]
    C4 -->|no| R
    L -->|no| C5{"method == STORED?"}
    C5 -->|no| X5["IOException: entry must be STORED to be mapped"]
    C5 -->|yes| OK["Entry as, getDataOffset, getSize"]
```

The returned `Entry` keeps the *requested* name (`as`) but the data range of the final target. `entries()`
enumerates all ZIP entries, keeps those that resolve (silently skipping the rest) and sorts by name — this is what
`--list` prints.

### 5b. Reading an entry

- `channel(Entry)` returns an anonymous `ReadableByteChannel` with its own `position`, reading via positional
  `FileChannel.read(dst, offset + position)` and clamping the buffer limit to the remaining entry size. It never
  moves or closes the shared archive channel; `close()` only marks the view closed.
- `extract(Entry)` copies the view to `Files.createTempFile("jinfer-tts-", suffix)` (suffix from the entry's
  extension, else `.bin`), deletes it on failure or if its size differs from `entry.size()`.

Models are **not** extracted: their weights are mapped in place from the archive file (section 6b). Companions
(voice, lexicon) are extracted because providers take them as `Path`s.

---

## 6. Loading sequences

### 6a. Model from a file

```mermaid
sequenceDiagram
    participant CLI as TtsCli
    participant M as Models
    participant P as ModelProvider
    CLI->>CLI: arena = Arenas.newCrossThread()
    CLI->>M: loadSpeech(Path, arena, companions)
    M->>M: open: FileChannel, GGUF.read header, rejectSplit
    M->>M: provider(gguf) by architecture, requireAccepted(companions)
    M->>P: loadSpeech(fc, gguf, path, arena, companions)
    P-->>M: SpeechSynthesisModel
    M-->>CLI: model
```

### 6b. Model from a `z://` archive entry

```mermaid
sequenceDiagram
    participant CLI as TtsCli
    participant SA as SelfArchive
    participant G as GGUF
    participant M as Models
    participant P as ModelProvider
    participant ML as ModelLoader
    CLI->>SA: entry(entryName(source))
    SA-->>CLI: Entry(name, offset, size)
    CLI->>SA: channel(entry)
    CLI->>G: GGUF.read(header channel)
    G-->>CLI: gguf, offsets relative to entry start
    CLI->>M: loadSpeech(archive.fileChannel(), gguf, entry.offset, entry.size, archive.path(), arena, companions)
    M->>M: validate range, rejectSplit, ModelLoader.requireComplete
    M->>M: provider(gguf), requireAccepted
    M->>P: loadSpeech(channel, gguf.at(baseOffset), path, arena, companions)
    P->>ML: map tensors in place from the archive channel
    Note over ML: refuses a tensor data offset that is not 4-byte aligned
    P-->>CLI: SpeechSynthesisModel
```

`gguf.at(baseOffset)` rebases the header's offsets onto the enclosing file, so the provider reads weights straight
from the executable. `archive.path()` is passed as the model path, which is where a port looks for sibling files.

Resource lifetime: the `Arena` from `Arenas.newCrossThread()` is `Arena.ofShared()` on the JVM and `Arena.ofAuto()`
in a native image (where `close` is a no-op); `Arenas.close` is the best-effort release in `finally`, after which the
extracted companion temp files are deleted.

---

## 7. Output modes

```mermaid
flowchart LR
    U["use model, options"] --> SO["speechOptions: NONE or SpeechOptions.speed"]
    SO --> Q{"mode"}
    Q -->|"--stream"| ST["AudioPlayer.stream model, text, options, printFirstAudio"]
    Q -->|"--play"| PL["model.speak text -> Media.Audio -> AudioPlayer.play"]
    Q -->|default| WR["write: model.speak, AudioCodec.wav, Files.write output"]
    WR --> RP["print path, audio seconds, elapsed, x realtime"]
```

```mermaid
sequenceDiagram
    participant CLI as TtsCli
    participant AP as AudioPlayer
    participant SSM as SpeechSynthesisModel
    participant OS as external player
    CLI->>AP: stream(model, text, options, firstAudio)
    AP->>SSM: sampleRate()
    AP->>SSM: speak(text, options, sink)
    loop each synthesized clip
        SSM-->>AP: Media.Audio clip
        AP->>OS: PCM pipe (aplay/ffplay) or one WAV clip per process (afplay / PowerShell)
        AP-->>CLI: firstAudio.run() on the first clip
    end
```

`--play` waits for the whole waveform (`speak` concatenates the clips via `Media.Audio.concat`) before playback;
`--stream` starts after the first chunk. Audio seconds in `write` are `pcm.length / channels / sampleRate`.

---

## 8. Build, packaging and resources

- `pom.xml`: `maven-shade-plugin` produces `target/jinfer-tts.jar`; properties `jinfer.image.name=jinfer-tts` and
  `jinfer.image.mainClass=com.qxotic.jinfer.tts.TtsCli` feed the shared native-image recipe.
- Profile `native`: `maven-resources-plugin` copies `lexicon.bin` from
  `${jinfer.models}/hf.co/remixerdec/Inflect-Nano-v2-GGUF` into the classes (when present) so Inflect can find its
  lexicon on the classpath; then `native-maven-plugin` builds the image. `Makefile` delegates `make native` to the
  repository root Makefile (`NATIVE_MODULE=jinfer-tts`), output in `bin/`.
- Self-archive packaging (README): `zip -0` the payload, append it to the executable, then `zip -A` to fix offsets.
  Entries must be `STORED`; model data must land 4-byte aligned.
- Tests: `OptionsTest`, `SelfArchiveTest` (in `src/test`).

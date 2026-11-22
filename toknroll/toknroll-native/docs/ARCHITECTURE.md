# toknroll-native — Architecture

`toknroll-native` (Java 25, GraalVM Native Image) builds **`libtoknroll`**: Tok'n'Roll as a
shared/static C library with the API of [`include/toknroll.h`](../include/toknroll.h), and the
Python package `toknroll` that binds it. The Java side (`com.qxotic.toknroll.libtoknroll`) is a
set of `@CEntryPoint`s behind a small C shim (`src/main/c`).

Dependencies: `toknroll-core`, `toknroll-gguf`, `toknroll-hf`, `toknroll-vector`, `jinfer-hub`
(source grammar, model store, fetch through curl), GraalVM `nativeimage` / `svm`.

| Class | Role |
|---|---|
| `Native` | The `toknroll__*` entry points (one per header function); handles over `Loaded` |
| `Sources` | `toknroll_tokenizer_load(source)`: resolve → sniff format → load (HF / GGUF / tiktoken) |
| `Stores` | The jinfer `ModelStore` the sources resolve through (cache root, offline mode) |
| `Errors` | Thread's last error (code + message in C thread-local storage) |
| `Escape` | Uninterruptible `CEntryPoint.ExceptionHandler`s for OOM / stack overflow |
| `Workers` / `Runs` | `toknroll_encode_batch` over C worker threads, texts shared out in runs |
| `WellFormed` | UTF-8 validation, so well-formed input is encoded without decoding |
| `Log` / `LogFinder` | `System.LoggerFinder` routing every log line to the host's `toknroll_log_fn` |
| `ShimConsole` | jinfer-hub `Console` through the shim's C helpers (no FFM in the image) |
| `Faults` | `-Dtoknroll.test.escape=<where>:<kind>` fault injection for the C tests |

## 1. Layers: C API → shim → entry points → core

```mermaid
classDiagram
    direction TB
    class toknroll_h {
        <<C header>>
        toknroll_version()
        toknroll_last_error(code)
        toknroll_log_set(fn, user)
        toknroll_configure(options, count)
        toknroll_tokenizer_load(source)
        toknroll_tokenizer_free(t)
        toknroll_encode(t, text, length, ...)
        toknroll_encode_batch(t, texts, ...)
        toknroll_decode(t, ids, count, ...)
        toknroll_vocabulary_size(t)
        toknroll_token_flags(t, id)
        toknroll_token / toknroll_id
    }
    class toknroll_shim_c {
        <<C>>
        isolate per process, attaches threads
    }
    class toknroll_isolate_c {
        <<C>>
        thread-local last error, worker pool, tty helpers
    }
    class Native {
        <<public final>>
        +handle(Tokenizer)$ ObjectHandle
        ~errorSet()$ toknroll__error_set
        ~lastError()$ toknroll__last_error
        ~logStart()$ toknroll__log_start
        ~load()$ toknroll__tokenizer_load
        ~free()$ toknroll__tokenizer_free
        ~encode()$ toknroll__encode
        ~encodeBatch()$ toknroll__encode_batch
        ~decode()$ toknroll__decode
        ~vocabularySize()$ toknroll__vocabulary_size
        ~tokenFlags()$ toknroll__token_flags
        ~token()$ toknroll__token
        ~id()$ toknroll__id
    }
    class Loaded {
        <<record>>
        Tokenizer tokenizer
        Specials specials
        Map~Integer,String~ texts
        ~of(Tokenizer)$ Loaded
        ~spelling(int id) String
        ~id(String spelling) int
    }
    class Buffers {
        <<private nested>>
        thread-reused byte / id buffers
    }
    class Tokenizer {
        <<core>>
    }
    class Specials {
        <<core>>
    }
    toknroll_h ..> toknroll_shim_c : implemented by
    toknroll_shim_c ..> Native : calls toknroll__*
    toknroll_shim_c ..> toknroll_isolate_c
    Native *-- Loaded
    Native *-- Buffers
    Loaded --> Tokenizer
    Loaded --> Specials
```

A `toknroll_tokenizer*` is an `ObjectHandles` word over a `Loaded`. Another library hosting
these symbols next to its own (libjinfer) obtains handles via `Native.handle(Tokenizer)`.

## 2. Loading a source

```mermaid
classDiagram
    direction LR
    class Sources {
        <<final>>
        ~load(String source)$ Tokenizer
        ~load(String source, ModelStore)$ Tokenizer
        ~locate(String source, ModelStore)$ Located
        ~sniff(Path file)$ Kind
    }
    class Located {
        <<record>>
        Path path
        boolean folder
        ModelRef repository
        String ggufFile
    }
    class Kind {
        <<enumeration>>
        GGUF
        JSON
        UNKNOWN
    }
    class Stores {
        <<final>>
        ~store()$ ModelStore
        ~offline()$ boolean
    }
    class ModelStore {
        <<jinfer-hub>>
    }
    class GGUFTokenizerLoader {
        <<toknroll-gguf>>
    }
    class HuggingFaceTokenizerLoader {
        <<toknroll-hf>>
    }
    Sources *-- Located
    Sources *-- Kind
    Sources ..> Stores
    Stores ..> ModelStore
    Sources ..> GGUFTokenizerLoader : GGUF (header only when remote)
    Sources ..> HuggingFaceTokenizerLoader : tokenizer.json / tiktoken.model
```

Source grammar is jinfer-hub's (local file/folder, `user/repo`, `user/repo/file.gguf`, …); the
cache is shared with libjinfer. Knobs: `-Dtoknroll.cache.root` / `TOKNROLL_CACHE_ROOT`,
`-Dtoknroll.offline=true` (or `jinfer.offline`).

## 3. Errors, escapes, batches and logging

```mermaid
classDiagram
    direction TB
    class Errors {
        <<final>>
        ~clear()$
        ~set(Throwable)$ int
        ~code(Throwable)$ int
        ~last()$ CCharPointer
        ~lastCode()$ int
        ~invalid(String)$ Failure
        ~notFound(String)$ Failure
    }
    class Failure {
        <<RuntimeException>>
        int code
    }
    class Escape {
        <<build-time initialized>>
        ~words(int kind)$ CCharPointer
    }
    class ExceptionHandler {
        <<CEntryPoint>>
    }
    class IntFailure
    class LongFailure
    class HandleFailure
    class StringFailure
    class VoidFailure
    class TaskFailure
    class Workers {
        <<final>>
        ~run(int tasks, IntConsumer task)$
        ~task()$ toknroll__task
    }
    class Runs {
        ~of(long[] starts, int first, int last, int k)$ int[]
    }
    class WellFormed {
        ~utf8(byte[], int, int)$ boolean
    }
    class Log {
        <<final>>
        ~start(given, Fn callback, user)$
        ~emit(int level, String)$
    }
    class Fn {
        <<CFunctionPointer>>
        invoke(level, message, userData)
    }
    class LogFinder {
        <<System.LoggerFinder service>>
    }
    class ShimConsole {
        <<jinfer-hub Console>>
        +isTerminal(fd) boolean
        +columns(fd) int
        +enableAnsi(fd) boolean
        +enableUtf8() boolean
    }
    class Faults {
        ~at(String where)$
    }
    Errors *-- Failure
    Escape *-- IntFailure
    Escape *-- LongFailure
    Escape *-- HandleFailure
    Escape *-- StringFailure
    Escape *-- VoidFailure
    Escape *-- TaskFailure
    ExceptionHandler <|.. IntFailure
    ExceptionHandler <|.. LongFailure
    ExceptionHandler <|.. HandleFailure
    ExceptionHandler <|.. StringFailure
    ExceptionHandler <|.. VoidFailure
    ExceptionHandler <|.. TaskFailure
    Workers ..> Runs
    Log *-- Fn
    LogFinder ..> Log : loggers
    Log ..> ShimConsole
    Native ..> Errors
    Native ..> Escape
    Native ..> Workers
    Native ..> WellFormed
    Native ..> Faults
```

* Every entry point catches what it can → `Errors.set` (thread's last error in C TLS, no
  allocation) and returns NULL / negative (`TOKNROLL_ERROR`, `EINVAL`, `ENOTFOUND`, `EUNSUPPORTED`).
* What no catch can see (OOM / stack overflow raised by native image) reaches the entry point's
  `Escape.*Failure` handler: uninterruptible, allocation-free, initialized at build time.
* `toknroll_encode_batch` splits the texts into runs (`Runs`) and runs them on C worker threads
  (`Workers`, pool in `toknroll_isolate.c`); in a forked child the batch runs on the calling thread.
* Logging: `<PROGRAM>_LOG` level, host callback or stderr; no `java.util.logging` in the image.

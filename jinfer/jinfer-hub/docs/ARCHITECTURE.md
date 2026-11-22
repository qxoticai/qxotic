# jinfer-hub — class architecture

Package `com.qxotic.jinfer.hub` (JPMS module `com.qxotic.jinfer.hub`, 14 top-level types, ~4700 lines). It is jinfer's
**model downloader and cache**: a model reference such as `unsloth/Qwen3.5-4B-GGUF:Q8_0` goes in, a local GGUF path comes
out, and only missing files are downloaded, from Hugging Face or ModelScope. It has **no inference dependency**: the pom
depends only on `json` (`com.qxotic.format.json`) for the listing APIs and `jinfer-fetch` (`com.qxotic.jinfer.fetch`) for
the HTTP GET. The only exported API is `ModelStore`, `ModelRef`, `ModelSource` and its two public implementations,
`RemoteFile`, `ModelNotFoundException`, `Console`, `FfmConsole` and `TerminalSupport`. `RepositorySource`, `Hub`, `Fetch` and
`RateLimit` are package-private.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` nested/owned; classes marked `<<external>>` live outside this module.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams): 2a public API and sources, 2b download machinery, 2c console SPI
3. [Model ref grammar](#3-model-ref-grammar)
4. [Resolution flow](#4-resolution-flow): 4a `resolve`, 4b `fetchPipeline`, 4c source fallback
5. [Cache layouts](#5-cache-layouts): flat store and the shared Hugging Face hub cache
6. [Download engine (`Fetch`)](#6-download-engine-fetch)
7. [Rate limits](#7-rate-limits)
8. [Console SPI and progress](#8-console-spi-and-progress)
9. [Configuration knobs](#9-configuration-knobs)

| Class | Role |
|---|---|
| `ModelStore` | Public façade. `standard()` / `of(root, sources...)`; `resolve`, `resolveAll`, `pullAll`, `select`, `find`, `cached`, `detached`, `remove`, `isRef`, `requireRef`. Owns cache layout, GGUF/quant selection, offline policy, source fallback. Nested `Policy` enum (`USE_CACHE`, `REFRESH`, `REDOWNLOAD`) and `Cached(ref, sizeBytes)` record |
| `ModelRef` | Record `(host, owner, repo, revision, path, quant)`: the pure grammar `[host/]owner/repo[@revision][/path][:quant]`, validation, URL refusal plus suggestion. Nested `Host` enum (`HF`, `MODELSCOPE`): the closed table of hosts |
| `ModelSource` | Transport seam interface: `supports(ref)`, `list(ref, dir)`, `fetch(ref, file, into)` |
| `RepositorySource` | Package-private `ModelSource` for one `ModelRef.Host`: listing (HF tree API / ModelScope files API), file URLs, credentials, `commitFor` (HF refs API) |
| `HuggingFaceSource` | Public final subclass for `Host.HF`; honors `HF_ENDPOINT` or an explicit `URI` |
| `ModelScopeSource` | Public final subclass for `Host.MODELSCOPE`; honors `MODELSCOPE_ENDPOINT` |
| `RemoteFile` | Record `(path, sizeBytes, sha256)` as a listing reports a file; `sha256` may be null |
| `ModelNotFoundException` | `IllegalArgumentException` carrying `source()`: the ref/path names nothing reachable |
| `Hub` | Package-private: reads and writes the shared Hugging Face hub cache (`models--owner--repo/blobs`, `snapshots/<commit>`, `refs/`) |
| `Fetch` | Package-private download engine: parallel ranged GETs, resume map, sha256 verify, atomic publish, file + in-JVM locks, redirects, stall guard, progress board. Nested `Board`, `Stall`, `Validator`, `Progress`, `HttpStatusException`, `InvalidTransfer`, `SingleStreamRequired`, `Sent`, `IOCall`, `StagedDownload` |
| `RateLimit` | Package-private: parses `Retry-After` / `RateLimit` / `RateLimit-Reset`, keeps a per-host hold every request waits out |
| `Console` | SPI interface (`uses` in module-info): `isTerminal`, `columns`, `enableAnsi`, `enableUtf8`, all defaulting to "no console" |
| `FfmConsole` | The registered `Console` provider: libc (`isatty`, `ioctl`) or kernel32 console APIs via the Foreign Function API |
| `TerminalSupport` | Static façade choosing a `Console` (given via `use`, else `ServiceLoader`, else none); used by `Fetch.Progress` and `jinfer-cli` |

---

## 1. Position in the jinfer stack

Nothing in the inference engine calls this module (stated in `ModelStore`'s javadoc): resolution happens in callers before
`Models.load`, so a library load given a path never touches the network.

```mermaid
flowchart TB
    subgraph callers["Callers (depend on jinfer-hub)"]
        CLI["jinfer-cli: Main, Hub, Terminal"]
        NAT["jinfer-native: libjinfer Model"]
        LC4J["jinfer-langchain4j: JinferChatModel, ..."]
        SAI["jinfer-spring-ai / -autoconfigure"]
        TK["jinfer-testkit"]
    end
    subgraph hub["jinfer-hub"]
        MS["ModelStore"]
        MR["ModelRef"]
        SRC["ModelSource / RepositorySource"]
        HUB["Hub (HF hub cache)"]
        F["Fetch + RateLimit"]
        TS["TerminalSupport / Console"]
    end
    subgraph fetch["jinfer-fetch"]
        T["Transport (JdkTransport | CurlTransport)"]
    end
    JSON["json: com.qxotic.format.json.Json"]
    CHAT["jinfer-chat: Models.load(path)"]

    callers -->|"resolve / resolveAll / requireRef"| MS
    callers -->|"path"| CHAT
    CLI --> TS
    MS --> MR
    MS --> SRC
    MS --> HUB
    SRC --> F
    HUB --> F
    SRC --> JSON
    F --> T
    F --> TS
```

---

## 2. Class diagrams

### 2a. Public API, sources and errors

```mermaid
classDiagram
    direction LR
    class ModelStore {
        <<final>>
        -Path root
        -boolean hubShare
        -List~ModelSource~ sources
        +standard()$ ModelStore
        +of(root, sources...)$ ModelStore
        +root() Path
        +isRef(String)$ boolean
        +requireRef(String)$
        +resolve(pathOrRef) Path
        +resolveAll(List) List
        +pullAll(List, force) List
        +select(ref) String
        +find(pathOrRef) Optional
        +cached() List
        +detached() List
        +remove(pathOrRef) boolean
        -resolveRef(ref, policy) Path
        -fetchRef(ref, policy) Path
        -fetchPipeline(ref, source, policy) Path
        ~select(ref, source) RemoteFile
        -byQuant(ref, folder, files) RemoteFile
        -cachedFile(ref) Path
        ~ofQuant(ref, names)$ List
    }
    class Policy {
        <<enum>>
        USE_CACHE
        REFRESH
        REDOWNLOAD
    }
    class Cached {
        <<record>>
        String ref
        long sizeBytes
    }
    class ModelRef {
        <<record>>
        String host, owner, repo
        String revision, path, quant
        +parse(String)$ ModelRef
        +repoId() String
        ~isRef(String)$ boolean
        ~notAReference(url)$ String
        ~suggestionFor(url)$ ModelRef
        ~revisionOrDefault() String
        ~cacheRepo() String
        ~repoUrl() String
    }
    class Host {
        <<enum>>
        HF
        MODELSCOPE
        name, defaultRevision
        tokenEnv, endpointEnv
        defaultBase, prefix, aliases
        base() String
    }
    class ModelSource {
        <<interface>>
        supports(ref) boolean
        list(ref, dir) List~RemoteFile~
        fetch(ref, file, into)
    }
    class RepositorySource {
        <<package-private>>
        -Host host
        -String base
        fileUrl(ref, revision, file) String
        commitFor(ref) String
        denied(ref)$ String
        headers(host)$ Map
    }
    class HuggingFaceSource { <<final>> }
    class ModelScopeSource { <<final>> }
    class RemoteFile {
        <<record>>
        String path
        long sizeBytes
        String sha256
    }
    class ModelNotFoundException {
        <<final>>
        -String source
        +source() String
    }
    class IllegalArgumentException { <<external>> }

    ModelStore *-- Policy
    ModelStore *-- Cached
    ModelRef *-- Host
    ModelStore --> "0..*" ModelSource : sources, tried in order
    ModelSource <|.. RepositorySource
    RepositorySource <|-- HuggingFaceSource
    RepositorySource <|-- ModelScopeSource
    RepositorySource --> Host
    ModelStore ..> HuggingFaceSource : standard()
    ModelStore ..> ModelScopeSource : standard()
    ModelStore ..> ModelRef : parse
    ModelSource ..> RemoteFile : list
    IllegalArgumentException <|-- ModelNotFoundException
    ModelStore ..> ModelNotFoundException : throws
    RepositorySource ..> ModelNotFoundException : 401/403/404
```

### 2b. Download machinery (package-private)

```mermaid
classDiagram
    direction LR
    class ModelStore { <<final>> }
    class RepositorySource { <<package-private>> }
    class Hub {
        <<final, package-private>>
        cache()$ Path
        cachedPath(ref, hubCache)$ Path
        repoFolder(ref, hubCache)$ Path
        cached(hubCache)$ List
        detached(hubCache)$ List
        fetchInto(ref, file, commit, hubCache, source, force)$ Path
        publicationKey(ref, hubCache)$ Path
        link(blob, dest)$ Path
        writeRef(refFile, commit)$
        remove(cached, hubCache)$ boolean
        isCommit(String)$ boolean
        isSha256(String)$ boolean
    }
    class Fetch {
        <<final, package-private>>
        getString(url, headers)$ String
        download(url, dest, label, size, sha256, headers)$
        locked(dest, IOCall)$ T
        replace(dest, size, sha256, StagedDownload)$
        matches(file, size, sha256)$ boolean
        publish(stage, dest)$
        send(uri, headers, timeout, budget)$ Response
        announce(message)$
        oneAtATime()$ boolean
        remainingBytes(dest, size)$ long
    }
    class Board { <<static nested>> }
    class Progress { <<static nested>> }
    class Stall { <<private nested>> }
    class Validator { <<private nested>> }
    class HttpStatusException {
        <<static nested>>
        int status
    }
    class InvalidTransfer { <<private nested>> }
    class SingleStreamRequired { <<private nested>> }
    class IOException { <<external>> }
    class RateLimit {
        <<final, package-private>>
        MAX_WAIT, BUDGET, LISTING_BUDGET
        requested(response, now)$ Duration
        hold(host, wait)$ boolean
        awaitClear(host, budget)$ long
        httpDate(String)$ long
    }
    class Transport {
        <<interface, external>>
        get(uri, headers, timeout) Response
        system()$ Transport
    }
    class Json { <<external>> }

    Fetch *-- Board
    Fetch *-- Progress
    Fetch *-- Stall
    Fetch *-- Validator
    Fetch *-- HttpStatusException
    Fetch *-- InvalidTransfer
    IOException <|-- HttpStatusException
    IOException <|-- InvalidTransfer
    InvalidTransfer <|-- SingleStreamRequired
    Board --> "0..*" Progress : rows
    Fetch --> RateLimit
    Fetch --> Transport : system().get
    RepositorySource --> Fetch : getString, download
    RepositorySource --> Json : parseMap, parseList
    Hub --> Fetch : replace, download, publish
    Hub --> RepositorySource : fileUrl, headers
    ModelStore --> Hub
    ModelStore --> Fetch : locked, replace, announce
```

### 2c. Console SPI

```mermaid
classDiagram
    direction LR
    class Console {
        <<interface>>
        isTerminal(fd) boolean
        columns(fd) int
        enableAnsi(fd) boolean
        enableUtf8() boolean
    }
    class FfmConsole {
        <<final>>
        ISATTY, IOCTL
        GET_STD_HANDLE, GET_CONSOLE_MODE ...
    }
    class TerminalSupport {
        <<final>>
        -Console console$
        +use(Console)$
        +isTerminal(fd)$ boolean
        +columns(fd)$ int
        +enableAnsi(fd)$ boolean
        +enableUtf8()$ boolean
    }
    class ServiceLoader { <<external>> }
    class Linker { <<external, java.lang.foreign>> }
    class Progress { <<Fetch.Progress>> }
    class Terminal { <<external, jinfer-cli>> }

    Console <|.. FfmConsole
    TerminalSupport --> Console : chosen
    TerminalSupport ..> ServiceLoader : load(Console)
    FfmConsole ..> Linker : downcallHandle
    Progress ..> TerminalSupport : enableAnsi(2), enableUtf8
    Terminal ..> TerminalSupport
```

SPI registration: `module-info.java` declares `uses com.qxotic.jinfer.hub.Console` and
`provides com.qxotic.jinfer.hub.Console with com.qxotic.jinfer.hub.FfmConsole`; the class path equivalent is
`src/main/resources/META-INF/services/com.qxotic.jinfer.hub.Console` (one line: `FfmConsole`). `Console`'s javadoc says a
native image leaves that service out and gives its own provider through `TerminalSupport.use`.

---

## 3. Model ref grammar

`[host/]owner/repo[@revision][/path][:quant]`: `/path` says where, `:quant` says which, `@revision` says when.

```mermaid
flowchart TB
    In["string"] --> URL{"contains ://"}
    URL -->|yes| Refuse["IllegalArgumentException<br/>notAReference: suggests a ref<br/>for a hub page URL"]
    URL -->|no| H{"first segment in Host table<br/>hf.co, huggingface.co, modelscope.cn ..."}
    H -->|yes| Strip["strip host"]
    H -->|no| Bare{"isBareRefShape:<br/>not a path shape, no dot in first segment,<br/>2+ segments, not owner/x.gguf"}
    Bare -->|no| Shape["IllegalArgumentException: shape message"]
    Bare -->|yes| Def["host = HF"]
    Strip --> Split["segments; quant = after last ':' of final segment;<br/>revision = after '@' in repo segment"]
    Def --> Split
    Split --> Rec["new ModelRef: every field one path component,<br/>no ':', '/', '\\', '..'; quant and a file path are not both allowed"]
```

- `ModelRef.isRef(s)` = host-qualified, **or** bare ref shape and no existing local file of that name (a file wins).
- `Host` rows carry `defaultRevision` (`main` / `master`), `tokenEnv` (`HF_TOKEN` / `MODELSCOPE_API_TOKEN`),
  `endpointEnv` (`HF_ENDPOINT` / `MODELSCOPE_ENDPOINT`), `defaultBase` and URL `prefix` (`""` / `/models`).
- `suggestionFor(url)` maps hub page URLs (`tree`, `blob`, `resolve`, ModelScope `file/view`, `files`, `summary`) back
  to a ref, turning a root `<model>-<QUANT>.gguf` into `:QUANT` via `quantOf`.

---

## 4. Resolution flow

### 4a. `ModelStore.resolve(pathOrRef)`

```mermaid
sequenceDiagram
    participant C as caller
    participant S as ModelStore
    participant R as ModelRef
    participant H as Hub
    participant F as Fetch
    C->>S: resolve(pathOrRef)
    S->>R: isRef(pathOrRef)
    alt not a ref
        S->>S: refuseUrl, localFile
        S-->>C: Path, or ModelNotFoundException
    else ref
        S->>R: parse
        S->>S: resolveRef(ref, USE_CACHE)
        S->>S: cachedFile(ref): own flat cache first
        S->>H: cachedPath(ref, Hub.cache()) then quant pick in snapshot
        alt cache hit
            S-->>C: cached Path (no network)
        else miss
            S->>S: requireOnlineFor (JINFER_OFFLINE / -Djinfer.offline)
            S->>F: locked(publication key or repoDir, fetchRef)
            F->>S: fetchRef(ref, policy)
            S-->>C: downloaded Path
        end
    end
```

`resolveAll` runs each element through the same path, concurrently on a pool of at most 4 threads (sequentially when
there is one argument or `Fetch.oneAtATime()`); the first failure cancels the rest. `pullAll(refs, force)` uses
`Policy.REFRESH` / `REDOWNLOAD`, which skip the plain cache hit and re-check upstream. `select(ref)` returns the chosen
repository-relative file without downloading, remembering a quant's answer in `<repo>/quants/<quant>` for offline use.

### 4b. `fetchPipeline(ref, source, policy)` — one source

```mermaid
flowchart TB
    A["fetchPipeline"] --> C["RepositorySource.commitFor(ref)<br/>HF only: /api/models/owner/repo/refs, or the ref itself if a commit"]
    C --> D["downloadRef = ref pinned to commit"]
    D --> Sel["select(downloadRef, source)<br/>list parent; file named? else list as folder; byQuant"]
    Sel --> Split{"name matches -NNNNN-of-NNNNN.gguf"}
    Split -->|yes| U["UnsupportedOperationException: merge with llama-gguf-split"]
    Split -->|no| Sh{"shared = hubShare and dest absent<br/>and RepositorySource and hf.co<br/>and sha256 valid and commit known"}
    Sh -->|yes| HF["Hub.fetchInto: blobs/sha256 + snapshots/commit/path link + refs/rev"]
    Sh -->|no| Ref{"policy == REFRESH and cached copy<br/>matches size + sha256"}
    Ref -->|yes| UpToDate["announce up to date, return cached"]
    Ref -->|no| W["requireWritable, requireDiskSpace 110%,<br/>tagCacheDirectory CACHEDIR.TAG"]
    W --> P{"policy"}
    P -->|USE_CACHE| SF["source.fetch(downloadRef, file, dest)"]
    P -->|REFRESH / REDOWNLOAD| RP["Fetch.replace(dest, size, sha256, stage -> source.fetch)"]
```

Selection rules (`byQuant` / `ofQuant`): only `.gguf` files that are not companions (`mmproj*`, `mtp*`) count as models;
one model and no quant means that file; otherwise the quant (explicit, else `DEFAULT_QUANT = "Q4_K_M"`, llama.cpp's
default) must match exactly one `-<QUANT>.gguf` / `.<QUANT>.gguf` file. None gives a `ModelNotFoundException` listing
`quants(...)`; several gives a copy-pasteable menu. A repository with only safetensors is refused before any download.

### 4c. Source fallback (`fetchRef`)

```mermaid
flowchart LR
    L["sources filtered by supports(ref)"] -->|empty| E["UncheckedIOException: no source serves host"]
    L --> I["for each source in order"]
    I --> T["fetchPipeline"]
    T -->|ok| Ok["return; INFO log if not the first"]
    T -->|"IOException / HttpStatusException / IAE caused by HttpStatusException"| N{"last source?"}
    N -->|no| Wn["WARNING log, next source"]
    Wn --> I
    N -->|yes| Th["throw: 401/403 as ModelNotFoundException,<br/>else UncheckedIOException or the IAE"]
    T -->|"other runtime exception"| P["propagates: the ref's fault"]
```

---

## 5. Cache layouts

```mermaid
flowchart TB
    subgraph own["Flat store: root = -Djinfer.models, JINFER_MODELS, or platform cache/jinfer"]
        O1["root/CACHEDIR.TAG"]
        O2["root/hf.co/owner/repo/file.gguf"]
        O3["root/modelscope.cn/owner/repo@rev/sub/file.gguf"]
        O4["root/host/owner/repo/quants/q4_k_m (last selection)"]
    end
    subgraph hubc["Shared HF hub cache: HF_HUB_CACHE, HF_HOME/hub, XDG_CACHE_HOME/huggingface/hub"]
        B["models--owner--repo/blobs/sha256"]
        S["models--owner--repo/snapshots/commit/path  (relative symlink, else hard link, else copy)"]
        RF["models--owner--repo/refs/main  (commit text)"]
    end
    S -->|links| B
    RF -->|names| S
```

- `hubShare` is true only when `root` equals the platform default by value **and** the hub cache is writable; then
  `hf.co` downloads are written into the hub cache (visible to llama.cpp and `hf download`). An explicit root opts out.
  ModelScope always uses the flat layout. The hub cache is **always read** (`cachedFile`, `cached()`).
- Directory names are matched case-insensitively (`ModelStore.anyCase`), so differently cased ids share one folder.
- `cached()` merges own entries and `Hub.cached` (own first, one per ref); `detached()` lists GGUFs in snapshots no ref
  names, counting shared blobs once. `remove` deletes a flat file and prunes empty folders, or calls `Hub.remove`, which
  deletes the blob only when no other snapshot links it, then stale refs, then an emptied repository folder.

---

## 6. Download engine (`Fetch`)

```mermaid
flowchart TB
    D["Fetch.download(url, dest, label, size, sha256, headers)"] --> Lk["locked(dest): per-path ReentrantLock<br/>+ FileLock in tmpdir/jinfer-locks-user/hash.lock"]
    Lk --> Ex{"dest exists"}
    Ex -->|yes| Done["return"]
    Ex -->|no| Tr["transfer: up to 3 attempts"]
    Tr --> Par{"size >= 64 MB"}
    Par -->|yes| PA["parallel: 32 MB chunks, pool of THREADS (4..8),<br/>positional writes into pre-sized dest.part,<br/>dest.part.map byte per chunk after force()"]
    PA -->|SingleStreamRequired| SQ
    PA --> V["verify sha256 over the whole file"]
    Par -->|no| SQ["sequential: resume with Range + If-Range,<br/>incremental digest"]
    V --> Pub["publish: ATOMIC_MOVE part to dest"]
    SQ --> Pub
    Pub --> Cl["cleanup .map / .etag"]
    Tr -->|InvalidTransfer| Disc["discard part, map, etag"]
    Tr -->|"4xx other than 408/429, disk full, interrupted"| Fail["throw"]
```

- **HTTP**: every request goes through `send` → `sendOnce` → `Transport.system().get(...)` (`jinfer-fetch`: JDK
  `HttpClient` on a JVM, `curl` in a native image). Redirects are followed by hand (max 5); `Authorization` is dropped
  when the origin changes, and HTTPS→HTTP downgrades are refused (`redirectTarget`).
- **Integrity**: `Validator` pins one strong ETag (stored in `.etag`) across workers and retries, else requires the
  sha256; `validateRange` checks `Content-Range`; `Accept-Encoding: identity` is forced.
- **Stall guard**: `Stall` registers each body stream with a daemon `jinfer-stall-guard` thread that closes a stream
  with no bytes for `jinfer.downloadStallSeconds` (default 60); the same value bounds the header wait.
- **Replace** (`Fetch.replace`): refresh downloads go to `dest.refresh.part`, are size- and sha256-checked, then
  atomically moved over the old entry, so a failed refresh keeps the published file.
- `getString` (listings) uses a 20 s header timeout, `RateLimit.LISTING_BUDGET` and a 64 MB body cap; a non-200 raises
  `HttpStatusException`, which `RepositorySource.get` turns into `ModelNotFoundException` (401/403/404) or a
  "limiting requests" `IOException` (ModelScope 403 with code `10013201001`).

```mermaid
sequenceDiagram
    participant S as ModelStore
    participant RS as RepositorySource
    participant F as Fetch
    participant RL as RateLimit
    participant T as Transport
    S->>RS: list(ref, dir)
    RS->>F: getString(api url, headers)
    F->>RL: awaitClear(host)
    F->>T: get(uri, headers, 20s)
    T-->>F: Response
    alt 429 or 503 with stated wait
        F->>RL: requested(response), hold(host, wait)
        F->>RL: awaitClear(host, budget)
        F->>T: get again
    end
    F-->>RS: JSON body
    RS-->>S: List of RemoteFile (via Json.parseList / parseMap)
    S->>RS: fetch(ref, file, dest)
    RS->>F: download(fileUrl, dest, name, size, sha256, headers)
    F->>T: GET resolve URL, 302 to CDN, ranged GETs
    F-->>RS: dest complete or absent
```

---

## 7. Rate limits

`RateLimit.requested` reads, in order, `Retry-After` (seconds or an RFC 1123 date parsed by hand in `httpDate`), the IETF
`RateLimit` structured field (`"policy";r=0;t=21`, only exhausted policies count) or its older `reset=` dictionary form,
then `RateLimit-Reset`. `Fetch.send` waits the asked time (or `5s << backoff`, capped at 60 s, when none is stated), capped
at `MAX_WAIT` (15 min) plus 250–1250 ms jitter, within a budget of `BUDGET` (30 min) for downloads or `LISTING_BUDGET`
(1 min) for listings. `hold(host, wait)` stores a process-wide per-host deadline in a `ConcurrentHashMap`, so every
chunk worker pauses together; only the caller that extended the hold logs and announces it.

---

## 8. Console SPI and progress

```mermaid
flowchart LR
    P["Fetch.Progress static init"] -->|"ENABLED: jinfer.progress / JINFER_PROGRESS"| B{"BOARD = ENABLED and no NO_COLOR<br/>and TERM != dumb and TerminalSupport.enableAnsi(2)"}
    B -->|yes| Board["Fetch.Board: live region on stderr,<br/>one row per transfer, jinfer-progress ticker ~10 fps"]
    B -->|"no, ENABLED"| Plain["PLAIN: a line per tenth on stderr"]
    P -->|"disabled"| Log["start/done lines to System.Logger jinfer.hub at INFO"]
    TS["TerminalSupport.console()"] -->|"use(given)"| G["given Console"]
    TS -->|"else ServiceLoader"| FC["FfmConsole"]
    TS -->|"else"| N["anonymous Console: no terminal"]
```

`FfmConsole` links symbols only when `FfmConsole.class.getModule().isNativeAccessEnabled()`; otherwise every probe
answers no. On POSIX it uses `isatty` and `ioctl(TIOCGWINSZ)` (`0x5413` Linux, `0x40087468` macOS); on Windows
`GetStdHandle`, `Get/SetConsoleMode` (virtual terminal processing), `GetConsoleScreenBufferInfo` (visible window width)
and `Get/SetConsoleOutputCP` (UTF-8, original restored by a `jinfer-console-restore` shutdown hook).
`TerminalSupport.columns` falls back to `$COLUMNS`, then 80. `native-image.properties` initializes `Fetch`,
`Fetch$Progress`, `FfmConsole` and `TerminalSupport` at run time and enables the Foreign API.

---

## 9. Configuration knobs

| Setting | Read by | Effect |
|---|---|---|
| `-Djinfer.models` > `JINFER_MODELS` > platform cache `/jinfer` | `ModelStore.ambientRoot` | Store root for `standard()` |
| `-Djinfer.offline` > `JINFER_OFFLINE` | `ModelStore.offlineSource` | Misses fail as `ModelNotFoundException`; pulls throw `IllegalStateException` |
| `-Djinfer.downloadThreads` > `JINFER_DOWNLOAD_THREADS` | `Fetch.threads` | Chunk workers (default clamp(cpus, 4, 8)); 1 also serializes `resolveAll` |
| `jinfer.downloadStallSeconds` | `Fetch.stallSeconds` | Stall guard and header timeout (default 60) |
| `-Djinfer.progress` > `JINFER_PROGRESS`, `NO_COLOR`, `TERM=dumb` | `Fetch.Progress` | Progress on/off and board vs plain lines |
| `JINFER_SKIP_DISK_CHECK=1` | `ModelStore.requireDiskSpace` | Skip the size + 10% free-space check |
| `HF_TOKEN`, `HF_TOKEN_PATH`, `HF_HOME/token`, `~/.cache/huggingface/token`; `MODELSCOPE_API_TOKEN` | `RepositorySource.headers` | `Authorization: Bearer` |
| `HF_ENDPOINT`, `MODELSCOPE_ENDPOINT` | `ModelRef.Host.base` | Mirror base URL |
| `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME` | `Hub.cache` | Location of the shared hub cache |

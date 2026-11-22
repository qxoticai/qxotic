# jinfer-cache — class architecture

Package `com.qxotic.jinfer.cache` (6 classes + `package-info`, ~2840 lines). It is jinfer's **prompt (KV) cache**:
one production façade, `PromptCache`, over two layers — **retained sessions** (the last N live conversation states,
continued in place when a prompt strictly extends one) and an optional **block layer** (content-addressed,
budget-bounded KV checkpoints in RAM, optionally backed by an on-disk catalog file that survives restarts). The module
knows nothing about any particular model: a model contributes only a `CheckpointCodec`, storage is a `CacheStore`, and
identity is a `ContentKey` seed.

Dependencies (`pom.xml`): `jinfer-core`, `jota-core`, `jota-memory`, `toknroll-core`. No `META-INF/services` SPI —
the module is a plain library; its only resource is
`META-INF/native-image/com.qxotic/jinfer-cache/native-image.properties` (`--initialize-at-run-time=…CacheStore`,
because the store owns a `Cleaner` whose daemon thread cannot survive image build). JPMS automatic module name
`com.qxotic.jinfer.cache`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` owns (composition; label `nested` = nested type); classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a overview, 2b `PromptCache` and its nested types, 2c block layer
3. [Construction: `PromptCache.of`](#3-construction-promptcacheof)
4. [Serving protocol: `serve`](#4-serving-protocol-serve)
5. [Block layer: resume and commit](#5-block-layer-resume-and-commit)
6. [Define / save / export and the catalog lifecycle](#6-define--save--export-and-the-catalog-lifecycle)
7. [Fingerprints and keys](#7-fingerprints-and-keys)
8. [JKVF file format (`FrozenBlocks`)](#8-jkvf-file-format-frozenblocks)
9. [Memory ownership](#9-memory-ownership)

| Class | Role |
|---|---|
| `PromptCache<S>` | Public façade, `AutoCloseable`. Nested `Options` (immutable config + the shared `contextCapacity` rule), `Tier` enum (`SESSION`/`BLOCKS`/`FRESH`), `Serving` interface (pass handle), `Pass<S,R>` functional interface, private `Live` (the `Serving` impl), `Sample` record. Owns the retained-session pool, the `BlockTree` and the `CacheStore` |
| `CachedSession<S>` | A live `ContextState` bound to its exact fingerprint stream; the block layer's *write handle* (commits each ingested span). Three modes: full, sessions-only (no tree), read-only (define-only serving). Also holds the optional *tail snapshot* used for coarse rewind |
| `BlockTree<S>` | The block layer's policy: prefix tree of variable-length KV blocks keyed by a chained SHA-256 (`BlockKey`), longest-prefix resume, commit with dedup, LRU-leaf eviction under a byte budget, frozen-base grafting, `freeze` / `appendTo`. Nested `BlockKey` record, inner `Block`, `Sample` record |
| `FrozenBlocks` | The on-disk artifact format **JKVF** v4: `createEmpty`, `open` (lazy mmap + validation), `describe`, `write`, crash-safe `append` (dual checksummed commit slots, file lock). Nested `Entry` and private `Commit` records |
| `CacheStore` | Blob allocator: one cross-thread arena per blob, deterministic `free`/`close`, exact `usedBytes()` for the budget, `Cleaner` backstop + `LeakWatch` |
| `Sha256` | Package-private helper: SHA-256 `MessageDigest` and digest → 4 little-endian longs |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph callers["Callers"]
        CE["jinfer-chat ChatEngine<br/>serve / define / save / export / sample"]
        NAT["jinfer-native libjinfer Model, State<br/>Options.contextCapacity"]
        CLI["jinfer-cli Options, Hub<br/>cacheOptions, FrozenBlocks.describe"]
        SPR["jinfer-spring-ai, tests, bench"]
    end
    subgraph cache["jinfer-cache"]
        PC["PromptCache"]
        CSn["CachedSession"]
        BT["BlockTree"]
        FB["FrozenBlocks"]
        ST["CacheStore"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel / ContextModel"]
        CTX["ContextState"]
        CC["CheckpointCodec"]
        CK["ContentKey"]
        BA["Batch"]
        AR["Arenas · LeakWatch · PerformanceCliff"]
    end
    SEED["jinfer-chat Models.modelSeed"]
    DISK[("catalog file JKVF")]

    CE --> PC
    NAT --> PC
    CLI --> PC
    CLI --> FB
    SPR --> CE
    SEED -.->|ContentKey seed| PC
    PC --> CSn --> BT --> FB --> DISK
    BT --> ST
    PC --> ST
    PC --> LM
    CSn --> LM
    BT --> CC
    CSn --> CC
    BT --> CTX
    PC --> BA
    ST --> AR
```

`ChatEngine` (jinfer-chat) is the production caller: it builds `PromptCache.of(loaded.model(), loaded.seed(), options)`,
runs every generation through `serve(prompt, pass, interrupt)` wiring `serving.tail(token)` /
`serving.tail(frame.row())` as the decoder's after-ingest hook, and exposes `define` / `export` / `save`. The CLI
reads catalog metadata with `FrozenBlocks.describe`; libjinfer and the CLI reuse
`PromptCache.Options.contextCapacity` as the one context-capacity rule.

---

## 2. Class diagrams

### 2a. Overview — relations between the package classes

```mermaid
classDiagram
    direction LR
    class AutoCloseable { <<interface, external>> }
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class ContextModel~C,W,S~ { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class ContentKey { <<record, external>> }
    class PromptCache~S~ { <<final>> }
    class CachedSession~S~ { <<final>> }
    class BlockTree~S~ { <<final>> }
    class FrozenBlocks { <<final>> }
    class CacheStore { <<final>> }
    class Sha256 { <<package-private>> }

    AutoCloseable <|.. PromptCache
    AutoCloseable <|.. CacheStore
    ContextModel <|-- LanguageModel
    PromptCache --> LanguageModel : model
    PromptCache --> CheckpointCodec : codec, nullable
    PromptCache *-- BlockTree : tree, nullable
    PromptCache *-- CacheStore : store, owned
    PromptCache o-- CachedSession : retained deque
    PromptCache ..> FrozenBlocks : createEmpty / open
    CachedSession --> ContextModel : model.ingest
    CachedSession --> ContextState : state
    CachedSession --> BlockTree : cache, nullable
    CachedSession ..> CheckpointCodec : tail snapshot
    BlockTree --> CheckpointCodec : capture / restore
    BlockTree --> CacheStore : allocate / free
    BlockTree --> FrozenBlocks : base, nullable
    BlockTree --> ContentKey : modelSeed
    FrozenBlocks --> ContentKey : modelSeed
    BlockTree ..> Sha256
    CachedSession ..> Sha256
    FrozenBlocks ..> BlockTree : BlockKey, chainRoot
```

### 2b. `PromptCache` and its nested types

```mermaid
classDiagram
    direction TB
    class PromptCache~S~ {
        <<final>>
        -model LanguageModel
        -tree BlockTree
        -codec CheckpointCodec
        -defineOnlyBlocks boolean
        -tailPerToken boolean
        -store CacheStore
        -writeBack Path
        -retained ArrayDeque~CachedSession~
        -retainedCapacity int
        -contextCapacity int
        +of(model, seed, options)$ PromptCache
        +serve(prompt, pass) R
        +serve(prompt, pass, prefillInterrupt) R
        +define(prompt)
        +save()
        +export(out)
        +blockCaching() boolean
        +contextCapacity() int
        +sample() Sample
        +treeStats() String
        +close()
    }
    class Options {
        <<static final>>
        +DEFAULT_CONTEXT_CAPACITY = 4096$
        +MAXIMUM_CONTEXT = -1$
        +DEFAULTS$
        +contextCapacity(requested, maximum, default)$ int
        +withCatalog(path, readOnly) Options
        +withBlockBudget(bytes) Options
        +withRetainedSessions(n) Options
        +withContextCapacity(n) Options
        +withMaxCheckpointOverhead(bytes) Options
    }
    class Tier {
        <<enumeration>>
        SESSION
        BLOCKS
        FRESH
    }
    class Serving {
        <<interface>>
        +tail(int token)
        +tail(Batch batch)
        +adopt(int[] tokens)
        +restored() int
        +tier() Tier
        +promptTime() Duration
        +prefillComplete() boolean
    }
    class Pass~S,R~ {
        <<interface>>
        +run(S state, Serving serving) R
    }
    class Live {
        <<private inner>>
        -session CachedSession
        -reply List~Long~
        -live boolean
        flushReply()
    }
    class Sample {
        <<record>>
        retainedSessions, sessionHits, stateAllocations
        sessionSnapshotBytes, blocks, bytes, budgetBytes
        blockHits, blockMisses, blockEvictions
        blockDiscards, blockRefusals
    }
    PromptCache *-- Options : nested
    PromptCache *-- Tier : nested
    PromptCache *-- Serving : nested
    PromptCache *-- Pass : nested
    PromptCache *-- Live : nested
    PromptCache *-- Sample : nested
    Serving <|.. Live
    Live --> CachedSession : session
    Live --> Tier
    PromptCache ..> Live : new per serve
    PromptCache ..> Pass : run
```

`Options.DEFAULTS` = 4 retained sessions, context capacity `0` (→ 4096 or the model's length if smaller), 2 GiB RAM
block budget, no catalog, 1 MiB max checkpoint overhead. `withBlockBudget(0)` disables the block layer.

### 2c. Block layer — `CachedSession`, `BlockTree`, `FrozenBlocks`, `CacheStore`

```mermaid
classDiagram
    direction LR
    class CachedSession~S~ {
        <<final>>
        -state S
        -cache BlockTree
        -commits boolean
        -tip Block
        -fp long[]
        -len int
        -snapshot MemorySegment
        -snapshotLen int
        +start(model, cache, state)$
        +resume(model, cache, state, prompt)$
        +resume(model, cache, state, prompt, maxPositions)$
        ~fresh(model, state)$
        ~fingerprints(batches)$ long[]
        +ingest(batches)
        +ingestGroups(groups)
        +step(token)
        +adopt(token)
        +adopt(int[] ingested)
        +detached() boolean
        +tokenIds() int[]
        ~streamIsStrictPrefixOf(req) boolean
        ~snapshotTail(codec)
        ~rewindToSnapshot(codec)
        ~dropSnapshot()
    }
    class BlockTree~S~ {
        <<final>>
        -codec CheckpointCodec
        -store CacheStore
        -budgetBytes long
        -modelSeed ContentKey
        -blocks Map~BlockKey,Block~
        -leaves Set~Block~
        -freshBlocks List~Block~
        -base FrozenBlocks
        -sentinel Block
        -detached Block
        ~chainRoot(seed)$ BlockKey
        ~resume(fp, len, state) Block
        ~commit(tip, fp, off, len, state) Block
        +freeze(Path)
        +appendTo(Path)
        +sample() Sample
        +stats() String
    }
    class BlockKey {
        <<record>>
        long a, b, c, d
    }
    class Block {
        <<inner>>
        key BlockKey
        parent Block
        from, to int
        mem MemorySegment
        children List~Block~
        lastUsed long
        live boolean
        frozen boolean
        frozenCrc int
    }
    class FrozenBlocks {
        <<final>>
        -file Path
        -modelSeed ContentKey
        -entries List~Entry~
        -commit Commit
        +createEmpty(file, seed)$
        +open(file, seed)$ FrozenBlocks
        +describe(file)$ String
        ~write(file, seed, entries)$
        ~append(fresh)
        ~contains(key) boolean
        ~crc32c(mem)$ int
        +blockCount() int
    }
    class Entry {
        <<record>>
        key, parentKey BlockKey
        from, to int
        offset long
        mem MemorySegment
        crc int
    }
    class CacheStore {
        <<final>>
        +inMemory()$ CacheStore
        +allocate(bytes) MemorySegment
        +free(blob)
        +usedBytes() long
        +close()
    }
    CachedSession --> BlockTree : cache
    CachedSession --> Block : tip
    BlockTree *-- BlockKey : nested
    BlockTree *-- Block : blocks, leaves
    Block --> Block : parent / children
    Block --> BlockKey
    BlockTree --> FrozenBlocks : base
    BlockTree --> CacheStore : store
    FrozenBlocks *-- Entry : entries
    Entry --> BlockKey
    BlockTree ..> Entry : freeze / appendTo build
```

---

## 3. Construction: `PromptCache.of`

`of` is the only constructor. It resolves the context capacity, reads the model's capabilities and picks one of three
shapes.

```mermaid
flowchart TD
    A["PromptCache.of model, seed, options"] --> B["capacity = Options.contextCapacity<br/>requested, model maxContextLength, 4096"]
    B --> C{"model.checkpointCodec present?"}
    C -- no --> D["PerformanceCliff.CACHE_SESSIONS_ONLY.report"]
    D --> E{"catalog configured?"}
    E -- yes --> X["IllegalArgumentException:<br/>catalog but no state codec"]
    E -- no --> S["sessions-only PromptCache<br/>tree = null, own CacheStore"]
    C -- yes --> F{"blockBudget &gt; 0 or catalog?"}
    F -- no --> S
    F -- yes --> G["store = CacheStore.inMemory"]
    G --> H["openCatalog: createEmpty if missing and read-write;<br/>missing read-only = error; FrozenBlocks.open"]
    H --> I["new BlockTree codec, store, budget, seed, base"]
    I --> J["overhead = codec.byteSize 0"]
    J --> K["defineOnlyBlocks = overhead &gt; maxCheckpointOverhead<br/>tailPerToken = overhead == 0<br/>writeBack = readOnly ? null : catalog"]
    K --> L["full PromptCache"]
```

The three derived modes:

| Mode | Condition | Effect |
|---|---|---|
| sessions-only | no codec, or budget 0 and no catalog | `tree == null`; sessions still continue in place; `define`/`export` throw |
| define-only blocks | `codec.byteSize(0) > maxCheckpointOverheadBytes` | serving restores but never commits (`commits=false`); only `define` writes; enables the *tail snapshot* coarse rewind |
| full | otherwise | every ingested span commits; `tailPerToken` when the codec carries no fixed residue (`byteSize(0) == 0`), else the reply commits as one block |

---

## 4. Serving protocol: `serve`

Cheapest source first: retained session → block prefix → fresh prefill. The prompt is ingested **before** the pass
runs; the pass only generates.

```mermaid
sequenceDiagram
    autonumber
    participant Caller as ChatEngine <<external>>
    participant PC as PromptCache
    participant CS as CachedSession
    participant BT as BlockTree
    participant M as LanguageModel <<external>>
    participant Pass as Pass lambda
    Caller->>PC: serve(prompt, pass, interrupt)
    PC->>CS: fingerprints(prompt)
    PC->>PC: checkFitsContext
    PC->>PC: sessionAcquire - longest retained stream strictly prefixing the prompt
    alt no session and defineOnlyBlocks
        PC->>CS: snapshotAcquire - snapshotIsStrictPrefixOf, rewindToSnapshot(codec)
    end
    alt no session hit
        PC->>PC: attach: recycled LRU state or model.newState
        PC->>CS: resume(model, tree, state, fp, len-1, !defineOnlyBlocks)
        CS->>BT: resume(fp, cap, state)
        BT-->>CS: matched tip, state.resumeAt(tip.to)
    end
    PC->>PC: groups = one per batch, split last token if tailPerToken
    opt defineOnlyBlocks and more than one group
        PC->>CS: ingestGroups(all but last)
        PC->>CS: snapshotTail(codec)
    end
    PC->>CS: ingestGroups(groups, fp, interrupt)
    loop each chunk not yet restored
        CS->>M: ingest(state, batch)
        CS->>BT: commit(tip, fp, off, len, state)
    end
    PC->>Pass: run(state, Live serving)
    loop decode
        Pass->>PC: serving.tail(token)
        PC->>CS: adopt(token) or buffer into reply
    end
    PC->>CS: flushReply: adopt(long[] reply) as one block
    alt prefill complete
        PC->>PC: release: retained.addLast, drop LRU past capacity
    else interrupted or exception
        PC->>PC: closeSession - state never serves again
    end
    PC-->>Caller: R
```

Key rules visible in the code:

* **The law** — every resume is capped at `fingerprints.length - 1` positions, so the last position is always
  re-ingested and logits are fresh; a cached answer equals a cold one.
* `Tier` is `SESSION` on a session/snapshot hit, `BLOCKS` when `restored > 0`, else `FRESH`.
* On a residue-free codec (`tailPerToken`), a final multi-token `LAST`-output prefill batch is split into
  `[ids..n-1] + [last]` so an identical future prompt can full-hit.
* `Live` expires when the pass returns: `tail`/`adopt` afterwards throw `IllegalStateException`.
* `release` discards a session whose stream length disagrees with `state.position()` (a caller that did not wire
  `tail`) and logs a warning — it must never poison future matches.
* With the pool full, a miss *recycles* the least-recent state (`dropSnapshot`, `reset`) instead of allocating:
  "the pool is the allocator".

---

## 5. Block layer: resume and commit

### 5a. `BlockTree.resume`

```mermaid
flowchart TD
    R0["resume fp, len, state"] --> R1["tip = sentinel"]
    R1 --> R2{"a child b with b.to &lt;= len and<br/>digest tip.key, fp from..to == b.key?<br/>longest span wins"}
    R2 -- yes --> R3["touch b; tip = b"] --> R2
    R2 -- no --> R4{"tip == sentinel?"}
    R4 -- yes --> R5["misses++; return sentinel"]
    R4 -- no --> R6["walk chain root to tip"]
    R6 --> R7{"block verified?<br/>frozen: CRC32C once, lazily"}
    R7 -- no --> R8["discard block and subtree; re-match from scratch"] --> R1
    R7 -- yes --> R9["codec.restore state, from, to, mem"] --> R7
    R9 --> R10["hits++; state.resumeAt tip.to; return tip"]
```

### 5b. `BlockTree.commit` (called by `CachedSession.commitSpan`)

```mermaid
flowchart TD
    C0["commit tip, fp, off, len, state"] --> C1{"tip.live?"}
    C1 -- no --> C2{"same key live again?"}
    C2 -- no --> C3["refusals++; return dead tip"]
    C2 -- yes --> C4["tip = live block"]
    C1 -- yes --> C5
    C4 --> C5["check state.position == tip.to + len"]
    C5 --> C6["key = SHA-256 of tip.key + fp span"]
    C6 --> C7{"key exists?"}
    C7 -- yes --> C8["dedup: touch, return existing"]
    C7 -- no --> C9{"ensureBudget codec.byteSize len<br/>evict LRU leaves, never tip"}
    C9 -- refused --> C10["refusals++; return detached"]
    C9 -- ok --> C11["mem = store.allocate; codec.capture state, from, to, mem"]
    C11 --> C12["new Block; register; leaves update;<br/>freshBlocks += if not in base"]
```

Eviction is **leaf-only LRU**, so every surviving chain stays contiguous; frozen (grafted) blocks are never leaves,
never evicted and never freed. A refused commit *detaches* the session (`CachedSession.detached()`); its state stays
valid for session reuse, it just stops reaching the shared tree. `define` turns a detach into an exception.

---

## 6. Define / save / export and the catalog lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant U as Caller <<external>>
    participant PC as PromptCache
    participant BT as BlockTree
    participant FB as FrozenBlocks
    participant F as catalog file
    Note over PC,F: boot - openCatalog
    PC->>FB: createEmpty(path, seed) if missing and read-write
    PC->>FB: open(path, seed) - mmap, validate header and seed, pick newest valid commit slot
    PC->>BT: new BlockTree(codec, store, budget, seed, base) - graft entries BFS as frozen blocks
    Note over U,BT: compile
    U->>PC: define(prompt)
    PC->>PC: newState, CachedSession.resume(cap total-1, commits=true)
    PC->>PC: ingestGroups(defineGroups) - interior batches, last position as its own single
    PC-->>U: throws if session detached (budget refused)
    U->>PC: save()
    PC->>BT: appendTo(writeBack)
    BT->>FB: append(fresh entries)
    FB->>F: lock, reopen disk view, write blobs and index, force, publish older slot, force
    U->>PC: export(out)
    PC->>BT: freeze(out) - refuses its own mounted file
    BT->>FB: write(out, seed, BFS entries with CRC32C)
```

`appendTo` recreates the whole file via `freeze` when the catalog was deleted mid-run, then remounts it. `append`
serializes writers JVM-wide (`synchronized (FrozenBlocks.class)`) and cross-process (`FileChannel.lock`), merges
against what is *already on disk*, and refuses with an `IOException` when a fresh block's parent is on neither side
(catalog rebuilt since mount). Each append leaves the previous index as dead bytes; `export` to a fresh file is the
compaction.

---

## 7. Fingerprints and keys

```mermaid
flowchart LR
    T["Batch.Input.Tokens"] -->|"token id as long"| FP["fingerprint stream long[]"]
    E["Batch.Input.Embeddings"] --> K{"contentKey?"}
    K -- yes --> CD["SHA-256 of key + positions"]
    K -- no --> RD["SHA-256 of raw FP32 row bytes + positions"]
    CD --> SP["digest i and 3 + GOLDEN * i"]
    RD --> SP --> FP
    SEED["ContentKey modelSeed"] --> ROOT["chainRoot = SHA-256 of seed digest"]
    ROOT --> BK["BlockKey = SHA-256 of parent key + span fingerprints"]
    FP --> BK
```

* `CachedSession.fingerprints(List<Batch>)` is the one fingerprint law, used both to match before ingest and to
  append during ingest. Media rows must be Panama-backed (`MemorySegment`) or fingerprinting throws.
* The chained `BlockKey` names the whole prefix; being cryptographic, matching trusts it as identity (no fingerprint
  storage, no collision handling).
* The model seed roots the chain, so two models (even sharing a tokenizer) can never match each other's blocks, and a
  catalog built under another seed fails `open` with an identity diff message.

---

## 8. JKVF file format (`FrozenBlocks`)

```mermaid
flowchart LR
    H["page 0: header 4 KiB<br/>magic JKVF, version 4, geometry,<br/>seed digest 32 B, CRC, seed description"]
    SA["page 1: commit slot A<br/>CMIT, generation, count, indexOffset, indexCrc, CRC"]
    SB["page 2: commit slot B"]
    KV["64-aligned KV blobs"]
    IX["index: 96 B entries, BFS parents-first<br/>key, parentKey, from, to, offset, len, crc"]
    H --> SA --> SB --> KV --> IX
```

`open` maps only up to the chosen commit's `committedLength` (automatic arena — unmapped by GC), validates every index
entry (span, alignment, parent precedes child, continuity, no duplicate keys) and leaves blob CRCs lazy
(`BlockTree.verified`). A torn publication falls back to the other slot. `describe` reports everything the file says
about itself, verifying all checksums, without needing a model.

---

## 9. Memory ownership

| Memory | Owner | Freed by |
|---|---|---|
| RAM block blobs | `CacheStore` (one `Arenas.newCrossThread()` arena per blob) | `BlockTree.unlink` → `store.free`; `PromptCache.close` → `store.close` |
| Frozen block blobs | mmap slices of `FrozenBlocks` (`Arena.ofAuto()`) | GC once unreachable; never counted against the budget |
| Retained states | `PromptCache.retained` | `closeSession` → `ContextState.close` |
| Tail snapshot | `CachedSession` (own cross-thread arena) | `dropSnapshot` (every session end path calls it) |

`CacheStore` frees deterministically (shared arenas) on the JVM; in a native image `Arenas.newCrossThread()` yields
automatic arenas (shared-arena support conflicts with the Vector API), so frees become GC-eventual while the byte
accounting stays exact. A `Cleaner` backstop sweeps an abandoned store, and `LeakWatch.arm` reports one that was never
closed. Everything in the package is single-threaded by contract (the generation worker); only
`CacheStore.usedBytes()` is meant to be read from other threads.

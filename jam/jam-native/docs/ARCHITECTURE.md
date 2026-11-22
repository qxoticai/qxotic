# jam-native — Architecture

`jam-native` (JPMS module `com.qxotic.jam.libjam`, provider id **`native`**, priority **1000**)
is the Java binding to **libjam**, the C/C++ (and Metal on Apple) matmul library built by CMake
from `src/`, `include/jam.h` and `src/kernels/`. The Java side is three classes; all the compute
is in native code.

| Class | Role |
|---|---|
| `NativeJAMProvider` | `JAM.Provider`: loads the library, probes symbols, creates `NativeJAM` |
| `NativeJAM` | `JAM` + `AutoCloseable`: owns one `jam_context*`, binds `jam_mm` (JNI or FFM), routes fan-outs to the host pool through an upcall |
| `NativeLoader` | Finds / extracts / `System.load`s the right `libjam` for the OS and arch; `config()` reads `-Djam.native.*` or `JAM_NATIVE_*` |

## 1. Class structure

```mermaid
classDiagram
    direction LR
    class JAM {
        <<interface>>
    }
    class Provider {
        <<interface JAM.Provider>>
    }
    class NativeJAMProvider {
        +id() "native"
        +priority() 1000
        +isAvailable() boolean
        +create(Parallel) JAM
        -warn(String)$
    }
    class NativeJAM {
        -long ctx
        -Parallel parallel
        -ReentrantReadWriteLock lock
        -Lock callLock
        -List~NativeJAM~ INSTANCES$
        -boolean SERIAL$
        -boolean METAL_RESIDENT$
        -boolean FFM$
        -MemorySegment PF_STUB$
        -int PACK_ABI_NATIVE$
        +create(Parallel)$ NativeJAM
        +mm(...) int
        +packSize(dtype, m, k) long
        +share(segment, immutable) boolean
        +close()
        ~probe()$ int
        ~lastError()$ String
        ~isa(String)$ int
        -mmJni(...)$ int
        -newJni(threads, maxIsa, parallel, user, metalResident)$ long
        -freeJni(ctx)$
        -mmFfm(...)$ int
        -pfUpcall(count, task, taskUser, user)$
        -unshare(addr)
    }
    class NativeLoader {
        <<final, package-private>>
        -boolean loaded$
        ~load()$
        ~override()$ String
        ~bundled()$ boolean
        ~config(prop, def)$ String
        ~os()$ String
        ~arch()$ String
        -cacheDir()$ Path
        -resource()$ String
    }
    class MemoryChecks {
        <<jam-core internal>>
    }
    class AutoCloseable {
        <<interface>>
    }
    Provider <|.. NativeJAMProvider
    JAM <|.. NativeJAM
    AutoCloseable <|.. NativeJAM
    NativeJAMProvider ..> NativeLoader : load(), override(), bundled()
    NativeJAMProvider ..> NativeJAM : probe(), create()
    NativeJAM ..> NativeLoader : load(), config()
    NativeJAM ..> MemoryChecks : requireNative / checkSegment / checkSpan
```

## 2. The Java ↔ C boundary

```mermaid
classDiagram
    direction LR
    class NativeJAM {
        <<Java>>
    }
    class jam_jni_c {
        <<C: jam_jni.c>>
        Java_..._newJni()
        Java_..._freeJni()
        Java_..._mmJni()
    }
    class jam_h {
        <<C API: include/jam.h>>
        jam_context_new(options)
        jam_context_free(ctx)
        jam_mm(ctx, w, wt, ldw, a, at, lda, r, rt, ldr, m, n, k)
        jam_pack_abi()
        jam_pack_size(ctx, wt, m, k)
        jam_share(ctx, p, bytes, flags)
        jam_unshare(ctx, p)
        jam_last_error(code)
        jam_isa_name(isa)
    }
    class kernels {
        <<C: src/kernels/*>>
        generic, sse3, avx2, avx_vnni
        avx512, avx512_vnni, bf16
        neon, dotprod, i8mm
    }
    class metal {
        <<ObjC++: jam_metal.mm>>
        simdgroup MMA prefill
    }
    class pool {
        <<C: jam_pool.c>>
        native pool (C hosts only)
    }
    NativeJAM --> jam_jni_c : JNI (default binding)
    NativeJAM --> jam_h : Panama downcalls (FFM binding, pack/share/error/isa always)
    jam_jni_c --> jam_h
    jam_h --> kernels : ISA dispatch, chosen once per context
    jam_h --> metal : Apple GPU, n >= 16 quant or any dense
    jam_h --> pool : only when no jam_parallel_fn is given
    jam_h ..> NativeJAM : jam_parallel_fn upcall (PF_STUB -> pfUpcall)
```

* `jam_mm` is reached through the **JNI shim** (`-Djam.native.binding=jni`, default) or a
  **Panama downcall** (`ffm`). Every other symbol (`jam_pack_size`, `jam_share`, `jam_unshare`,
  `jam_last_error`, `jam_isa_name`, `jam_pack_abi`) is always a Panama downcall.
* All symbols are bound during `NativeJAM` class initialization, so a stale library that lacks
  a symbol fails in `NativeJAMProvider.isAvailable()` (warn + fallback), not at the first matmul.

## 3. Host-driven threading (no native thread inside the JVM)

`NativeJAM.create(parallel)` passes `parallel.width()` as the context's slot count and an upcall
stub (`PF_STUB`) as `jam_parallel_fn`. The `user` value is the instance's index in `INSTANCES`,
so the static upcall can find the right host pool.

```mermaid
sequenceDiagram
    participant Caller
    participant NJ as NativeJAM.mm
    participant C as libjam jam_mm
    participant UP as pfUpcall (static)
    participant Pool as JAM.Parallel (host)
    participant T as jam_task_fn
    Caller->>NJ: mm(w, a, r, ...)
    NJ->>NJ: requireNative + checkSegment (bounds)
    NJ->>NJ: callLock.lock() (write lock if SERIAL)
    NJ->>C: mmJni / mmFfm (raw addresses)
    C->>UP: parallel(count, task, taskUser, user=index)
    UP->>Pool: run(<= 4 x width slices, (s, slot) -> ...)
    Pool->>T: runTask(begin, end, slot) via downcall
    T-->>Pool: done
    Pool-->>UP: all slices done
    UP-->>C: return (a Throwable is parked in UPCALL_FAILURE)
    C-->>NJ: status
    NJ->>NJ: rethrow parked failure if any
    NJ->>NJ: unlock + reachabilityFence(w, a, r)
    NJ-->>Caller: status
```

## 4. Concurrency and lifecycle

```mermaid
stateDiagram-v2
    [*] --> Open : create(parallel) / jam_context_new via newJni
    Open --> Open : mm() / packSize() / share()
    Open --> Closed : close() — takes write lock, waits for in-flight calls, freeJni
    Closed --> Closed : close() idempotent
    Closed --> [*]
    note right of Open
      SERIAL=true (default): mm holds the WRITE lock,
      concurrent calls queue FIFO.
      SERIAL=false: mm holds the READ lock,
      contention surfaces EBUSY from the native guard.
    end note
```

`share(segment, immutable)` calls `jam_share` and registers `jam_unshare` as a cleanup on the
segment's own scope (through a view-only `Arena` returned by `shareScope`), so a share ends when the
segment's arena closes; after `close()` the context has already released it.

## 5. Library loading

```mermaid
flowchart TD
    A[NativeLoader.load] --> B{"-Djam.native.library.path / JAM_NATIVE_LIBRARY_PATH set?"}
    B -- yes --> C[System.load override]
    B -- no --> D{"bundled resource /com/qxotic/jam/native/os-arch/libjam.*"}
    D -- missing --> E["System.loadLibrary(jam) (dev fallback)"]
    D -- present --> F[read lib.sha256]
    F --> G["cacheDir/sha256/libname exists?"]
    G -- no --> H[copy to temp .part then ATOMIC_MOVE]
    G -- yes --> I[System.load cached copy]
    H --> I
```

Cache directory: `$XDG_CACHE_HOME` or `~/.cache` (Linux), `~/Library/Caches` (macOS),
`%LOCALAPPDATA%` (Windows), each `/jinfer/jam`; `java.io.tmpdir/jinfer/jam` as a last resort.

## 6. Configuration knobs

| Property / env | Default | Effect |
|---|---|---|
| `jam.native.library.path` / `JAM_NATIVE_LIBRARY_PATH` | — | load this file instead of the bundled one |
| `jam.native.isa` / `JAM_NATIVE_ISA` | auto | cap the ISA by its `jam_isa_name` (`avx2`, `i8mm`, …); an unknown name throws |
| `jam.native.binding` / `JAM_NATIVE_BINDING` | `jni` | `ffm` for Panama downcall to `jam_mm` |
| `jam.native.serial` / `JAM_NATIVE_SERIAL` | `true` | `false` = surface `EBUSY` on contention |
| `jam.native.metal.resident` | `true` | keep Metal shares wired (macOS 15+) |
| `jam.native.disabled` | `false` | drop the provider in `JAM.providers()` |

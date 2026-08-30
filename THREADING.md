# Threading in jinfer and jam: audit and refactor plan

Branch `jam-vector-zen5-prefill`, 2026-08-29.
Five parallel read-only audits (host pool and its 82 callers, jinfer-kernels, jam-core/vector/scalar, jam-native C and JNI, engine/server/adapters/models/docs) feed this plan.
Every claim below carries a `file:line` from those audits.

## 1. The law we want

One pool, one budget, one knob, one contract:

- `jinfer.threads` participants (the caller plus `threads-1` workers) run every parallel region in the process.
- A region body receives `(index, slot)`: `index` is the work item, `slot` in `[0, width)` is the participant running it.
- Per-participant scratch is a `slot`-indexed table sized once by `width`, owned by the object that does the work (a `JAM` instance, a model `State`), never by a thread.
- The host offers its pool to every backend; the CPU backends in this repo use it and own no threads.
  A backend may run its own threads instead, under one rule: while `mm` runs it may use up to `width()` cores and the host's workers are quiescent; when `mm` returns the backend's workers are quiescent.
  Two pools spinning on one machine were measured worse in every configuration tried (7 t/s pinned decode, -30% prefill unpinned), so this rule is the whole interface, and it is checked by a thread census.
- `JAM.mm` is called from outside any region, one call at a time per instance; the host guarantees it, the backend asserts it.

Everything in the plan is either a deletion that gets closer to that law or the one addition (`slot`) that lets the deletions happen.

## 2. What the audits found

### 2.1 Correctness defects (fix regardless of the refactor)

| # | Defect | Evidence |
|---|---|---|
| D1 | libjam's per-worker repack scratch is sized by `availableProcessors` but indexed by the host's task index, unchecked, in 15 kernel sites. `-Djinfer.threads` above the logical CPU count (or any host with a larger `width()`) reads past `kq_repack[]` and writes through garbage pointers. | `NativeJAM.java:299`, `jam.c:421-425`, `jam_kernels_q4k_avx512.c:165,295,...` |
| D2 | `BandGemm` sweeps spin on `packed` with no failure escape; both real hosts keep running a region after a task throws, so one exception in a pack item wedges every other participant forever. | `BandGemm.java:279-301` vs `Gemm.java:89-95,147-150` (scalar has the escape) |
| D3 | `VectorSupport.host` is a process-wide static overwritten by every `VectorJAM` constructor, before its availability check. A second instance (tests do this) silently demotes the first to `INLINE`. | `VectorSupport.java:147`, `VectorJAM.java:47` |
| D4 | Lock-order inversion: backends hold `mmLock` across the host region (`mmLock -> Parallel.LOCK`) while a `mm` issued from inside a region takes `Parallel.LOCK -> mmLock`. Not reachable today only because no caller nests; nothing documents or asserts it. | `VectorJAM.java:97-114`, `ScalarJAM.java:67-75`, `Parallel.java:57-73` |
| D5 | `Parallel.Region.work` catches per chunk but counts the whole chunk finished; `ParallelTest.failuresPropagate...` asserts `done == 63` and is green only for thread counts where the chunk math lands right (fails at 3 or 5 P-cores). | `Parallel.java:117-126`, `ParallelTest.java:74-94` |
| D6 | (withdrawn in Phase 4) `ChatEngine.prepare` runs under a shared read lock beside a generation. The audit called the overlap fictional because a media projection queues on the pool; but a text prepare never touches the pool, and langchain4j's contract (tested: `startingAStreamDoesNotWaitForTheRunningGeneration`) is that `chat()` enqueues a stream without waiting for the reply in flight. The read/write split is what delivers that; it stays, with the reason now written at the field. | `ChatEngine.java:78-86` |
| D7 | `nativeQ4Decode` lacks the `NATIVE != null` guard its two siblings have, so on aarch64 with jam-native disabled Q4_0 decode lands on jam-scalar instead of the Vector API floor. | `MatMul.java:297-299` |
| D8 | Exceptions thrown inside a native fan-out propagate out of an FFM upcall stub, which terminates the VM. | `NativeJAM.java:220-229`, `Parallel.java:70` |

### 2.2 Dead code and dead configuration

| What | Where | Lines |
|---|---|---|
| libjam internal pthread pool, spin/condvar barrier, pinning, P/E-core and cgroup topology, `nthreads_bw`, capped fan, `jam_run_bw` | `jam_pool.c`, `jam_cpu.c/.h`, `jam.c:249-252,443-460,488-500` | ~500 C |
| Env vars `JAM_THREADS`, `JAM_NATIVE_THREADS`, `JAM_POOL`, `JAM_SPIN` (read only by the dead pool) and the `[jam] cpu: ... pinned=yes` banner that lies on the Java path | `jam.c:35-38,246,449-451`, `jam_pool.c:109-113` | |
| `createJni` (declared, never called) and `cfg.pool` (always NULL) | `jam_jni.c:9-20`, `NativeJAM.java:175` | |
| Stack-requant special case whose rationale ("concurrent jam_mm calls") is unreachable behind the busy guard; 10 KB frame on every call | `jam.c:544-560` | |
| `VectorSupport.parallelFor/parallelForEach/parallelChunks` (chunking on top of the pool's chunking; `parallelChunks` degenerates to one index per task) | `VectorSupport.java:308-353` | ~50 |
| jam-vector `Scratch` free list (ConcurrentHashMap of queues, 64 KiB size classes) serving exactly one buffer acquired under a lock | `Scratch.java:37-44,82-91` | ~25 |
| Five copies of "4 tasks per worker" | `Parallel.java:28`, `VectorSupport.java:154`, `BandGemm.java:105`, `Gemm.java:42`, `RowGemm.java:33` | |
| `PerformanceCliff.DECODE_CONTENTION` (zero call sites, describes the deleted SpinPool) | `PerformanceCliff.java:50-55` | |
| JFR `decodeThreads` always equal to `computeThreads` | `Telemetry.java:190-191`, `RuntimeEvent.java:39,43`, `EventSchemaTest.java:80` | |
| `Server.requestExecutor(int threads)` ignores `threads`; `--threads` is an admission limit documented as "HTTP handler threads" | `Server.java:302-304`, `Options.java:630`, `docs/jinfer/index.md:51` | |
| Stale docs: `JAM_VECTOR_THREADS`, `-Djam.threads`, `-Djam.<provider>.threads`, "each provider owns its workers", `SpinPool`, "decode pool", `jinfer.computeThreads/decodeThreads`, ForkJoin comments | `jam/README.md:50,144-154`, `docs/jam/index.md:102-121`, `jam.h:14-34,146-157`, `jinfer-bench/README.md:59,110-112`, `MatMul.java:67`, `jam-vector/Scratch.java:47-59`, `RuntimeFlags.java:64-66` | |

### 2.3 The same problem solved many ways

- **A per-participant slot.** `JAM.Parallel` promises "the task index doubles as a per-worker scratch slot" (`JAM.java:50-52`) but jinfer's `Parallel` hands out item indices, so every consumer fakes a slot: jam-vector and jam-scalar submit exactly `min(items, width)` tasks that each drain a private `AtomicInteger` (`BandGemm.java:263-272`, `Gemm.java:81-104`, `RowGemm.java:87-121`, `Gemv.java:45-56`); `FlashAttention` partitions `nParts` by hand (`FlashAttention.java:1655-1658`); `NativeJAM.pfUpcall` slices `n` into `width` pieces (`NativeJAM.java:224-228`); kernels fall back to `ThreadLocal` (`FlashAttention.java:113`, `MatMul.java:1685,1777`).
- **Pack-then-compute dependencies inside one region**, implemented three times with a spin barrier (`BandGemm.java:301`, `Gemm.java:147`, `RowGemm.java:115`), once without the failure escape (D2), all against `forLoop`'s own contract ("independent and non-blocking", `Parallel.java:51-52`).
- **Per-thread buffers**: `ThreadLocal` rooted in immortal daemon workers (`FlashAttention.Buffers` with `Arena.ofAuto` segments, `NVFP4_SCRATCH`, `TQ_SCRATCH`), slot-indexed state-owned scratch (`FlashAttention.DecodeScratch`, GDN `head*headDim`), two different jam `Scratch` classes (vector: free list + growable slot array with a lost-buffer race at `Scratch.java:67-74`; scalar: fixed `Slot[width]` frozen at construction, AIOOBE if `width` ever grows). jam-vector's own javadoc records the `ThreadLocal` leak as the lesson learned (`Scratch.java:13-17`); jinfer-kernels still does it.
- **Confined-arena workarounds** in three flavours: hoist scalars to a `float[]` (`Qwen35.java:523-528`), stage into a heap array and bulk copy (`VisionPreprocess.java:95-97`), allocate cross-thread arenas (`Gemma4Audio.java:81`).
- **Region inflation**: `Moe.dispatch` opens `2 x numExperts` regions per layer (`Moe.java:203-226`); `GptOss` ropes Q and K in two regions where every other port uses one (`GptOss.java:195,212`); `Gemma4Conformer` runs silu and saxpy over the same rows in two regions (`:253,256`, `:360,372`); the RoPE row loop is copied 8 times across ports.
- **Engine-level policy drift**: langchain4j prepares on the caller thread, spring-ai inside the stream lambda (`JinferStreamingChatModel.java:94` vs `JinferChatModel.java:474-481`); the two speech models advertise parallel requests that serialize on the pool (`JinferSpeechModel.java:40-42,94`); three immortal helper threads (`Sse.java:96`, `Fetch.java:139,283`).

### 2.4 What is already right (do not touch)

- All nine model modules use `Parallel.forLoop` and nothing else: no executors, no locks, no atomics (46 call sites).
- jinfer-kernels has no synchronization at all beyond disjoint output ranges plus the region barrier.
- `RuntimeState.enter()` with `tryLock` and a loud `ConcurrentModificationException`.
- The server's admission gate, single generation worker, and queue-full telemetry.
- `ChatEngine`'s stream driver and `cacheSnapshot` publication.
- `Parallel`'s lazy daemon workers with spin-then-park; there is no need for a shutdown API.

## 2.5 Main versus the branch: is one pool the right long-term call?

What `main` has: five independent pools and six knobs.
`Parallel.COMPUTE_POOL` and `DECODE_POOL` (two ForkJoin pools, `jinfer.computeThreads` / `jinfer.decodeThreads`), `SpinPool` (a third pool for the single decode submitter, `jinfer.decodeSpin`, with `PerformanceCliff.DECODE_CONTENTION` when a second submitter falls back to ForkJoin), jam-vector's own ForkJoin pool (`jam.vector.threads` / `jam.threads` / `JAM_VECTOR_THREADS`), and libjam's pthreads (`JAM_NATIVE_THREADS`, `JAM_POOL`, `JAM_SPIN`, pinning).
Each pool is sized to the machine, so any two of them active in the same token oversubscribe it; measured in this session as the pinned-decode collapse (7 t/s) and the 10% prefill loss on exact-size masks.

What the branch has: one pool, one knob, backends own no threads, `JAM.Parallel` is the seam.
Measured against main on the same JIT: Q8_0 pp512 700 -> 969 (native), 246 -> 426 (vector); tg32 unchanged at 27.2 on every backend; 2T decode 18.0 -> 21.2.

Verdict: the direction is right and this plan keeps it.
Three of the branch's mechanisms are not right long-term and the plan replaces them; two are debatable and the plan keeps them with the reason stated.

Replaced:

1. Backends that spin inside a task waiting for other tasks of the same region (`BandGemm.java:301`, `Gemm.java:147`, `RowGemm.java:115`).
   This only works if the host runs exactly `width` tasks concurrently and never abandons one; it is the source of D2 and the reason the backends bypass the pool's chunking with private drains.
   Phase 2 makes dependencies two regions.
2. `JAM.Parallel`'s "task index doubles as a scratch slot" (`JAM.java:50-52`), which the host cannot honor and every consumer works around.
   Phase 0 gives the body a real `slot`.
3. The static hosts (`VectorSupport.host`, D3) and the `availableProcessors`-sized C scratch (D1): per-instance state and `width`-sized slots in Phases 1 and 2.

Kept:

4. One process-wide `Parallel` with regions serialized by a lock.
   Main's ForkJoin pools let two sessions' regions interleave; the branch makes the second wait.
   Throughput is the same (every region already uses every core) and latency variance is lower, so serialization stays, with a fair lock.
   If jinfer ever needs per-engine thread budgets, the migration is mechanical: `Parallel` becomes an instance carried by `RuntimeState`, and the 82 static call sites read it from the state they already hold.
   No user needs that today, so it is not in this plan.
5. libjam driven through an FFM upcall instead of its own pthreads.
   Cost is one upcall per fan-out phase plus one downcall per slice, measured at decode parity with llama.cpp; the benefit is that the C library has no threading policy at all, which is what makes Phase 1's deletion possible.
   The corollary is that the upcall must keep handing libjam a bounded number of slices (`4 x width`, dynamic in the host), never one downcall per row; Phase 1 says so explicitly.

## 3. The refactor

Ordered so every phase leaves the tree green and benchmarkable, and later phases only delete.

### Phase 0: contract (jinfer-core, jam-core)

Done on the branch (uncommitted, 2026-08-29): `Parallel` is an instance API.
`Parallel.of(width)` creates a pool (workers start on the first region, `close()` stops them, a closed pool runs inline), `Parallel.shared()` is the process-wide pool sized by `RuntimeFlags.THREADS`, and the static `forLoop`/`threads()` are that instance's, so the 82 call sites are unchanged.
Regions are per instance (own lock, own workers, own nesting detection), so a backend or a test can own a pool without touching the shared one.
The chunk-level catch became a per-index catch (D5), the lock is fair, and `ParallelTest` covers two independent pools, close, and failure counting at widths 3, 5 and 16.
Measured neutral: Q8_0 16T pp512 935 native / 433 vector, 4T tg32 21.5.
Remaining items of this phase:

1. (done) `loop(int count, Body body)` where `Body { void run(int index, int slot); }`; the `IntConsumer` overloads stay; `JAM.Parallel` has the same `Body` with a default `IntConsumer` overload, and `MatMul.HOST` / the jam-scalar test `BenchPool` pass the slot through.
   The submitter runs with slot 0, worker `k` with slot `k + 1`; an inline loop (single element, width 1, nested) runs on the caller's own slot.
   Contract: within one loop, no two live iterations share a slot; two submitters of one pool each see their own slot 0, which is why slot-indexed scratch is owned by the loop's caller.
   Failure rule: a throwing body ends the loop early (what has not started is skipped, the first failure is rethrown once every participant has stopped); an inline loop stops at the throw like a plain loop.
   The earlier "every index still runs" rule was dropped after a 2^32-element test showed a region could not be ended once it had failed.
   Three bugs found by the tests: a worker starting after the first region was published skipped it, an interrupted worker spun forever because `park` returned at once, and `int` chunk claims wrapped for ranges ending near `Integer.MAX_VALUE` (claims are now `long` offsets).
   Tests: `ParallelTest` (50 cases: construction, ranges, slots, nesting, exceptions, concurrent submitters, close, wake-ups, interrupts, visibility, balance, dispatch cost) and `ParallelFuzzTest` (seeded random widths, ranges, spinning/throwing/nesting bodies, one and many submitters, against an oracle; `-Djinfer.test.seed`, `-Djinfer.test.fuzzRounds`); 20/20 repeated runs and 6 seeds x 400 rounds green.
   Round two added: integer-limit ranges, a 2^32 span, `StackOverflowError` on a worker, close from inside a region, concurrent submitters on a closed pool, width 64, a virtual-thread submitter, the shared pool inside an own region, another pool's worker submitting, sleeping bodies, 300k tiny regions, chunk-stable slots, early stop after a failure.
   Documented limit: two pools whose regions submit to each other at the same time deadlock (two locks in both orders).
   Round three (2026-08-30): the last region's body (and its captures) was retained until the next loop, `close()` racing the first loop could leave a parked worker behind, a throwable escaping the per-index handler would have hung the region (accounting is now in a `finally`, and a worker records what escapes instead of dying), the failure record is a CAS, own pools name their workers `jinfer-pN-k`, `toString()`, `requireNonNull` on every body, static `forLoop(start, end, Body)`, and the native-image run-time list names `Parallel$Shared` (the bench image builds and runs).
   Polish pass (2026-08-30): one allocation per region - the region's counters are `volatile` fields on `Region` driven by `VarHandle`s (no `AtomicLong`/`AtomicReference`), and a region carries the `IntConsumer` or the `Body` directly (no wrapper lambda); measured 64 bytes and ~0.7 us per empty region at width 4, ~4 us at width 16, workers allocate nothing; a test pins both.
   `ParallelLeakTest` (10 cases) runs every API path under a watchdog that dumps the pool's threads on a hang, with a thread census around `close()` in ten states, a collectability check for closed pools, capture-retention checks for live and shared pools, a 100k-region heap bound, stuck-worker checks after failed regions, lock release after failure, adversarial bodies (throwing on the submitter only, on workers only, first/last index, interrupting everyone, closing the pool from inside, nested failures, sleeping past the spin budget, the shared pool busy elsewhere) and 8 submitters hammering failing loops.
2. (done) per-index catch, fair lock.
3. `Parallel.inRegion()` (package-private or `jinfer` internal) for the assertion in MatMul (Phase 3).
4. `JAM.Parallel`: same `(index, slot)` body, `width()` documented as immutable for the instance's lifetime, and the sentence "a kernel never calls `forLoop` from inside a task, and the host never calls `mm` from inside a region" replaces the current "task index doubles as a slot" paragraph.
   The javadoc also states the opt-out: a backend may ignore the offered `Parallel` and run its own threads under the quiescence rule of section 1, and why the executor is the default (the two-pool measurements).
5. jam-core gains `Slots<T>`: `T[] sized by width`, `T get(slot)` with a lazy factory.
   That is the one shared scratch idiom for both Java backends.
6. `RuntimeEvent`: one `threads` field.
   Delete `PerformanceCliff.DECODE_CONTENTION`.

Verification: `ParallelTest` extended with a slot-uniqueness test (each slot busy at most once at a time) and a failure test that is thread-count independent; run at `-Djinfer.threads=3` and `=5`.

### Phase 1: jam-native owns nothing (C + JNI)

Done on the branch (uncommitted, 2026-08-30).
`jam_cpu.c/.h` (topology, P/E tiers, cgroups, pinning, 275 lines) deleted; `jam_pool.c` is a 110-line spin-then-park pool for C-only hosts, sized only by `jam_config.nthreads` (0 = online CPUs); `jam_run_bw`, `nthreads_bw`, the capped fan, the `[jam] cpu:` banner, the stack-requant case, `createJni` and the `JAM_THREADS`/`JAM_NATIVE_THREADS`/`JAM_POOL`/`JAM_SPIN` env vars are gone.
`jam_run` wraps every host task in a guard: a `tid >= nthreads` is refused and the call returns `EINVAL` (D1 closed; the kernels' unchecked `repack[tid]` is now safe by contract and by check).
`NativeJAM.create(parallel)` creates the context with `parallel.width()` as `nthreads` and keys the instance by the opaque `pool` handle jam passes back, so the upcall finds its own pool (no static host, D3 for native closed); the upcall slices `n` into at most `4 x width` pieces and passes the task's slot as `tid`; an exception inside a task is parked in a thread-local and rethrown by `mm` (D8 closed).
Tests: `HostPoolTest` (a real pool matches INLINE bit for bit on prefill and decode shapes at widths 2/4/8 with every slot used, a lying pool passing `tid` 7 at width 2 gets `EINVAL`, a throwing pool surfaces its exception from `mm`, instances are independent); `jam_test` 3261/3261; a running 16-thread prefill shows only `jinfer-*` threads in the JVM.
Numbers: Q8_0 16T pp512 997, tg32 27.2, 4T native decode 21.4; Q4_K_M 16T pp512 1087, 4T 34.3, 16T 40.9 with `kq.nativeDecode`; jam_bench own-pool gemv unchanged.

1. libjam keeps a minimal pool for C-only hosts (tests, benches, `jam.h` users without Java): one spin-then-park barrier, an explicit `nthreads`, nothing else.
   Delete `jam_cpu.c`, `jam_cpu.h` (topology, P/E tiers, cgroups, pinning), `nthreads_bw`, `jam_pool_parallel_for_capped`, `jam_run_bw`, the cpu banner, the `JAM_THREADS`/`JAM_NATIVE_THREADS`/`JAM_POOL`/`JAM_SPIN` env vars, `createJni`, `cfg.pool`.
   `jam_pool.c` shrinks to ~60 lines; `jam_run` is the only fan-out; a context has either `parallel_for` or the pool, and `nthreads` is also the number of `tid` slots.
2. The Java path always supplies `parallel_for` (as today), so no pthread is ever created inside a JVM.
3. D1: `createPfJni(host.width(), stub)`; `ensure_kquant` sizes `kq_repack[slots]` and `jam_mm` returns `EINVAL` if a `tid >= slots` ever arrives (cheap check in `jam_run`'s task wrapper, not in 15 kernels).
   `pfUpcall` keeps slicing `n` into at most `4 x width` contiguous slices (one downcall each, balanced dynamically by the host) and passes the host `slot` as `tid`; `tid` therefore no longer depends on how many slices there are.
4. Delete the stack-requant special case; `n == 1` uses `ensure_qscratch` like `jam.c:818` already does.
5. D8: `pfUpcall` catches `Throwable`, stores it in a field, returns; `NativeJAM.mm` rethrows after the downcall returns.
   The C task wrapper skips remaining tasks once `failed` is set.

Verification: `jam_test` (3261 checks), `jam_bench` at 1/4/16 threads, jinfer-bench Q8_0 and Q4_K_M pp512/tg32 at 2/4/16 threads against the numbers in `BENCHMARKS.md` and this session (Q4_K_M 4T tg32 34.5 native, 16T pp512 1011).

### Phase 2: Java backends on `(index, slot)`

Done on the branch (uncommitted, 2026-08-30).
Numbers (16T, Q8_0 2.6B, vanilla flags): pp512 Graal vector 436 / scalar 292 / native 997 (before: 430 / 292-300 / 997), C2 vector 365 / scalar 212 (before 359-367 / 205-300 run-dependent); Q4_K_M Graal vector 427, native 1065.
KernelBench Q8_0 2048x2048: vector unchanged; scalar Graal n=512 689 median (before ~700), n=64 370-385 median / 505-516 best (before 478): the `RowGemm` path lost the overlap of the reduce items with the last bands (32 band items over 16 slots at k=2048, so the tail is one item long and the reduce now waits for it); e2e pp512 does not use that path and is unchanged, so it stays as the price of the spin-free design.
jam-vector: `Scratch(parallel)` carries the pool (no `VectorSupport.host` static, D3 closed), a `perSlot[width]` panel table (`local(slot, need)`) and one grown `packed` buffer (the size-classed free list is gone); `VectorSupport.parallelFor/parallelForEach` deleted, `parallelChunks(parallel, count, body)` is the one helper left for the register-tile kernels; `BandGemm.run` is two regions (pack all tiles, then sweep all panels), no counter, no spin, no failure flag (D2 closed by construction); `Plan` takes the width.
jam-scalar: `Gemm` packs a token block's k-blocks in one region and sweeps its row groups in the next; `RowGemm` runs the band x split items, then one reduce per token; `Gemv` submits `ceil(m/16)` strips; every body takes its scratch from `scratch.slot(slot)`; the private `AtomicInteger` drains, `packed`/`done`/`failed` flags and the per-worker constants are gone.
jam-core ships `TestPool` in a test-jar (was jam-scalar's `BenchPool`), rethrowing the original exception like `Parallel`; `VectorKernelTest` runs under a 4-wide pool and has a throwing-dequant case (exception surfaces, the scratch is reusable); both `KernelBench`es take `-Djam.threads`.
Tests: jam-core 5, jam-scalar 8, jam-vector 31 (parity 21), jam-native 25x2, jinfer-core and jinfer-kernels suites green.

1. jam-vector: `VectorSupport.host` static becomes an instance field on a `Ctx(Parallel, Scratch)` that already rides every kernel signature (the `Scratch` parameter becomes `Ctx`).
   Delete `parallelFor/parallelForEach/parallelChunks`; kernels call `ctx.parallel().forLoop(items, ...)` directly with real item counts and use `slot` for `acquireLocal`.
   `Scratch` becomes `Slots<MemorySegment>` plus one monotonically grown `packedA` field; the free list goes.
2. `BandGemm`: pack region, then sweep region.
   Two barriers per gemm instead of one spin protocol; deletes the `packed` counter, the spin at `:301`, and closes D2 by construction.
   If KernelBench shows the second barrier costs more than 1% at n=64 (the sensitive shape), fall back to keeping one region but with the scalar backend's `failed` escape; the plan's default is two regions.
3. jam-scalar: `Gemm` packs each token block's k-blocks in a region, then row groups in a region; `RowGemm` bands then reduce; `Gemv` submits `ceil(m/16)` items and lets the pool chunk.
   Delete the three private `AtomicInteger next` drains, `packed`/`done` flags, `failed` flags, `GROUPS_PER_WORKER`/`ITEMS_PER_WORKER`.
   `Scratch` becomes `Slots<Slot>` + panels; the frozen-width AIOOBE disappears because `width` is now immutable by contract.
4. D4: each backend keeps exactly one `ReentrantLock` around `mm` (two engines in one JVM share the static `MatMul.NATIVE/VECTOR/SCALAR`), documented with the lock order, and `MatMul.mm` asserts `!Parallel.inRegion()` (Phase 3).
   `ScalarJAM`'s lock becomes fair like the other two; no reason for the difference exists.
5. jam-vector tests get a real pool: move `BenchPool` from jam-scalar's test tree to jam-core's test-jar and add a "task throws" test for both backends.

Verification: `VectorKernelTest`, `ScalarJamTest`, `JamBackendParityTest` under `BenchPool.of(4)`; `KernelBench` Q8_0 2048x512x2048 and n=64 on C2 and Graal, compared to 590/700 and 350/478 GMAC/s (scalar) and jam-vector's 16T pp512 426 t/s.

### Phase 3: jinfer-kernels

Done on the branch (uncommitted, 2026-08-30), except two items deliberately left:
`MatMul.run` is one `cells`-indexed region for every shape (the four arms are gone; the tiny cutoff applies in-place too; the in-place staging array stays, a per-chunk write-back is unsafe when the result aliases an operand); the dots take a `slot` and index per-slot NVFP4/ternary scratch tables (the two `ThreadLocal`s are gone); `MatMul.mm` refuses a call from inside a region of the shared pool (`Parallel.inRegion()`, tested) so the backend-lock/pool-lock inversion cannot happen; D7 fixed; `HOST`'s class-init-order dependency is stated.
`FlashAttention.Buffers` are one per slot of the shared pool, indexed by the region's slot (the `ThreadLocal` rooted in workers is gone); the table is process-wide and sized by the thread budget, a per-state scratch would need the prefill API to carry it through 15 call sites (ponytail note in the code). `FlashAttention` joined the native-image run-time list.
Left out: `Moe.dispatch` in two regions (the file is under Alfonso's own edit) and the fused row loops in GptOss/Gemma4Conformer (a region costs ~4 us at width 16, a few dozen per token is under 0.1 ms of a 30 ms token: not worth touching model code).
Verified: jinfer-core + jinfer-kernels suites (new `refusesACallFromInsideARegion`), the whole jinfer unit reactor, e2e 2.6B Q8_0 963/27.2, 8B-A1B Q8_0 1152/42.4, Q4_K_M 1057/44.0, 8B-A1B 4T floor decode 33.9.

1. `MatMul.run`: one `cells`-indexed region for all four arms; in-place uses a state-owned staging buffer and per-chunk copy-back instead of `new float[cells]` plus a serial writeback.
   `TINY_MATVEC_ELEMS` applies to both in-place and not.
   D7 one-token fix.
   `MatMul.mm` asserts `!Parallel.inRegion()`.
   `HOST` moves next to `load()` with the class-init order stated in one comment.
2. `FlashAttention.Buffers` leaves `ThreadLocal`: it becomes part of `DecodeScratch` (renamed `Scratch`), one per `State`, indexed by `slot`.
   `flashDecode` partitions by `slot` directly: `nParts` is `min(threads, range / blockSize + 1)` as today, partials indexed `slot * nHeads + h`, and the region submits `nParts * nHeads` items.
   `NVFP4_SCRATCH`/`TQ_SCRATCH`: the dots take the slot from the enclosing region body and index a `Slots<float[]>` on the caller's scratch.
3. `Moe.dispatch`: two regions per layer (gather all experts, scatter all experts) with `(expert, row)` decoded from the CSR offsets, instead of `2 x numExperts`.
4. Fuse the obvious pairs: `GptOss` Q+K rope, `Gemma4Conformer` silu+saxpy and glu+silu.
   Extract `Rope.applyRows` into jinfer-kernels and delete the 8 port-local copies (this is a de-duplication the threading work makes safe, not a threading change; it can trail).
5. Confined-arena staging: one helper `Parallel.gather(count, into float[])`-style idiom is not worth adding; standardize on the cross-thread arena the four Gemma4/Lfm2 encoders already use and delete the heap-staging copies in `VisionPreprocess`, `Lfm2VisionPreprocess`, `Lfm2Vision`.

Verification: `MatMulTest`, `MatMulJamParityTest`, `FlashAttentionTest`, `KernelSelectionTest`; jinfer-bench tg32 and pp512 on the 2.6B and the 8B-A1B MoE (region count per token is the metric for item 3; `SpinProbe` gives the per-region cost).

### Phase 4: engine and adapters

Done on the branch (uncommitted, 2026-08-30), with two items withdrawn after the tests spoke:
the engine's read/write lock stays (D6 above), and the speech adapters keep admitting requests in parallel (their tests pin the overlap; the compute interleaves on the shared pool, which the javadoc now says instead of promising a speed-up).
Landed: `Server.requestExecutor()` has no dead parameter and `--threads` is described as the admission limit it is; the SSE reaper is one thread per server, owned and interrupted by its `Running`; Fetch's progress ticker and stall guard exit when idle and restart on the next transfer (no immortal helper threads); the JFR runtime event carries one `threads` field.

1. `ChatEngine`: delete `lifecycle`; `prepare`, `prepareRaw`, `encode` take `lock`.
   One sentence of contract: one thread computes on an engine at a time.
   `MediaEncodingCache`: `synchronized` only on `sample()`.
2. langchain4j `prepare` moves inside the `stream` lambda, matching spring-ai.
3. Speech models: plain fair `ReentrantLock` like their sibling models; docs say "requests queue".
4. `--threads` becomes `--max-concurrent-requests`; `requestExecutor()` loses its parameter; `ServerExecutorTest` asserts the admission limit instead.
5. `Sse` reaper owned by `Running` (started in `Running`, stopped in `close()`), or replaced by a per-stream timeout on the handler executor; either is a deletion of the process-wide `AtomicBoolean` guard.
6. `Fetch`'s two immortal threads become one `ScheduledExecutorService` owned by the download, closed with it.

Verification: `jinfer-chat`, `jinfer-server` test suites; the server hardening harness from memory (`jinfer-server-hardening`); a two-client SSE run against `jinfer serve`.

### Phase 5: documentation sweep

Done (2026-08-30): `jinfer/ARCHITECTURE.md` has a Threading section (pool, slot, engine locks, server worker, driver), `docs/jinfer/index.md` names `-Djinfer.threads` as the only thread knob, `jinfer-hub/README.md` states the 4 x N connection fan-out, and the jam docs were rewritten in Phase 1. Tests added with Phase 4: `StallGuardTest` checks the stall guard leaves when nothing is watched, `ServerIntegrationTest.eachServerOwnsItsReaper` checks one reaper per server and none after close.

`jam/README.md`, `docs/jam/index.md`, `jam.h`, `jinfer-bench/README.md`, `jinfer/ARCHITECTURE.md` (gets the one-paragraph threading model: pool, slot, engine lock, server worker), `docs/jinfer/*.md` (the knob list: `jinfer.threads`, `jinfer.chat.streamQueueCapacity`, `jinfer.mediaCacheMB`, `jinfer.downloadThreads`, `jinfer.downloadStallSeconds`, `jinfer.decodeBlockSize`; and the 4x8 download fan-out).

## 4. What this deletes and what it adds

Deleted: libjam topology/pinning/bandwidth-capped fan (~440 of ~500 C lines; a 60-line pool stays for C-only hosts), four env vars, `jam_run_bw`, `createJni`, stack requant, `VectorSupport` chunk helpers, jam-vector free list, three spin-barrier protocols with their counters and flags, five copies of one constant, both `ThreadLocal` scratch families, `MatMul.run`'s four arms, `2N` MoE regions, the engine's second lock, the media-cache compute monitor, one JFR field, one dead cliff, one dead CLI parameter, and every stale sentence in the docs.

Added: the `slot` argument (one interface method, one int per region), `Slots<T>` (~20 lines), `tests/jam_threads.c` (~30 lines), `Parallel.inRegion()`, and a slot-uniqueness test.

## 5. Performance expectations and risks

- Decode and prefill numbers should not move; the kernels' inner loops are untouched.
  Measured today (2.6B, Graal, vanilla flags): Q8_0 pp512 969/426/300 (native/vector/scalar), tg32 27.2; Q4_K_M pp512 1011/426/300, tg32 44.2, and 34.5 at 4T with native decode.
- Expected small wins: fewer regions per token (MoE, fused loops), stable slot-to-participant affinity restoring the L2 reuse `Scratch.acquireLocal` was designed around, no `new float[cells]` per in-place matmul, no `ThreadLocal` lookups in the dot kernels.
- The one risk is Phase 2's pack/sweep split into two regions (one extra barrier per gemm, ~2-5 us).
  The n=64 KernelBench shape on C2 is the canary; the fallback is stated in the phase.
- C2's racy tier-3 profile under 16 threads (see memory `jam-scalar-autovector-kernels`) is unchanged by any of this: the scalar backend's run-to-run variance is a JIT property, not a threading one.

## 6. Out of scope, noted

- jota GPU backends keep their own executors; they are a different runtime.
- `Rope.applyRows` de-duplication and the confined-arena staging cleanup are safe follow-ups, not prerequisites.
- Pinning: measured in this session as a 10% prefill loss on exact-size masks on every backend; it stays out of the code.

## 7. Follow-ups landed after the plan (2026-08-30)

- Q5_K vector prefill: the 2.6B Q5_K_M ran at 211 or 432 t/s from run to run (3 of 10 slow), with the page cache, the backend routing and the machine load ruled out; the KernelBench never showed it because it measures one shape in a fresh JVM. `Q5KKernel.dequantizeRow` carried two `ByteVector`s through a loop phi (`qh0 = qh0.lanewise(LSHR, 2)`, `c == 0 ? qh0 : qh1`), the one K-quant dequant written that way; rewritten after `Q6KKernel`'s shape (fresh loads, literal shift counts through an inlined helper): 12 of 12 runs at 410-433 since. Law: never carry a Vector API value across a loop iteration or a conditional in a dequant; load it again.
- Q6_K native decode: the 2-bit plane extraction used 12 shift/and ops and two `inserti64x4` per half; now one `broadcast_i64x4` and one variable 16-bit shift per pair (`sllv`/`srlv` with per-half counts, masked to bits 4-5). Single-thread gemv 19.1 -> 23-26 GB/s; e2e 2.6B Q6_K 4T 24.7 -> 26.0 (llama.cpp 27.6), 2T 18.0, 8B-A1B Q6_K 4T 37.6 -> 38.4 (42.9). The remaining gap is the per-32 activation scale (llama.cpp's Q8_K per-256 scale lets the whole super-block accumulate in int32, ours applies a float scale per pair dot): a requant-format change for the K-quant path, not a kernel edit.
- Native image (2026-08-30): the jam-vector Q8_0/Q4_0/MXFP4 dequants read `ByteVector`s straight off the mapped weight segment; under native-image that load miscompiles into a general-protection fault (the JIT is fine), while every K-quant dequant reads through `VectorSupport.vectorSegment(w)` and works. All three now read through the absolute-address route. Two red herrings on the way: `@AlwaysInline` on `storeScaled` (needed for image performance, innocent) and the byte->float `castShape` (innocent). Image, vector backend, 16T pp512: Q8_0 crash -> 165, Q4_K_M 166. AOT results (native / floor decode) in the table of the session log: pp512 250-328 vs JIT 750-1160, tg32 16T 24-37 vs JIT 27-51: the image is still 3-4x behind the JIT on prefill and 10-30% on decode, the segment-load finding of the first session.
- Why the image was 3-4x behind the JIT (2026-08-30, resolved): a symbolized, frame-pointer image (`-H:-DeleteLocalSymbols -H:+PreserveFramePointer`, temporary) under `perf` showed 42% of prefill in the Vector API's generic fallbacks (`VectorSupport.ternaryOp`, `FloatVector$$L..load`, `Float512Vector.bOp`) called from `FlashAttention.lambda$prefill$0`, and `Convert.f16BitsToF32` as its own symbol under `dotQK`/`flashDecode` (a `FloatVector` returned across a call = boxed). native-image expands the Vector API per method within a budget: the prefill lambda had every tile `@AlwaysInline`d into it, blew the budget, and nothing in it expanded. Fix: the memory-in/memory-out tiles (`qkTile*`, `pvTile*`, `decodeF16Run`) are `@NeverInline` (each its own expansion unit; a stub of `com.oracle.svm.core.NeverInline` joined the annotations module) and `Convert.f16BitsToF32` is `@AlwaysInline`. Image, 2.6B Q8_0 16T: pp512 270 -> 911, tg32 23.8 -> 27.2 (JIT 993 / 26.9). Law: in the image, a Vector API kernel is a NOT-inlined method that takes and returns memory; never a helper that returns a vector, never one giant fused method.
- Chunk policy (2026-08-30): the branch's fixed 4-chunks-per-participant claims cost the memory-bound gemv 3% against main's one static band per worker (Q4_K_M 16T tg32 44.2 vs 45.2-45.9, bisected with `-Djinfer.chunks`: 1 -> 45.3 but straggler-fragile, 2 -> 45.3, 4 -> 44.3, 16 -> 42.6), while the band-gemm prefill needs every panel balanced individually (one band per participant cost it 10%). Guided (shrinking) claims sat in between with more variance. Resolution: `loop` claims half-bands (`size / (width * 2)`), and coarse items - jam-vector panels, tile chunks, jam-scalar groups and bands, native slices - use the new `forEach` (`JAM.Parallel.forEach`, one index per claim). Q4_K_M tg32 44.4-45.2, 8B-A1B 42.8, prefill unchanged. Also: `PerformanceCliff.NESTED_REGION` reports once when a loop runs inline because it was nested; the pool tests bound dispatch at 20 us per region.


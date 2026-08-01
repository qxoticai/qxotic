/**
 * jinfer's model APIs, and the ONE arena-ownership contract every one of them obeys.
 *
 * <h2>The contract</h2>
 *
 * Java has no {@code free}, and the FFM arenas jinfer allocates from are not garbage: a
 * mis-scoped arena is a leak, a double free, or a SIGSEGV. So there is exactly one rule, and every
 * API in jinfer is a spelling of it:
 *
 * <blockquote><b>Whoever creates an arena frees it.</b></blockquote>
 *
 * <h2>The three flavours, in every family</h2>
 *
 * Both {@link com.qxotic.jinfer.Model} and {@link com.qxotic.jinfer.SpeechModel} offer the same
 * three, with the same meanings. Only the parameter lists differ, because a generative state is
 * sized by context and batch and a speech state is not:
 *
 * <ul>
 *   <li><b>OWNED</b> ({@code newState(...)}, no arena) - the state creates an internal {@code
 *       ofShared} arena and {@code close()} frees it. The default: reach for it unless you have a
 *       reason not to.
 *   <li><b>BORROWED</b> ({@code newState(..., arena)}) - the state allocates from YOUR arena and
 *       {@code close()} never touches it. Close yours after your last call, never before.
 *   <li><b>ADOPTED</b> ({@code newState(..., arena, true)}) - the state takes your arena over and
 *       frees it, co-tenants (weights, say) included. For deliberately fusing one lifetime; adopt
 *       only when nothing in that arena outlives the state.
 * </ul>
 *
 * A non-closeable arena ({@code ofAuto}, {@code global}) may be adopted: owning it just means
 * there is nothing to free eagerly, and close stays a valid no-op on the memory.
 *
 * <h2>Which arena to create</h2>
 *
 * <ul>
 *   <li>{@code ofShared} - states, and weights a long-lived owner holds. Deterministic close, and
 *       usable from more than one thread, which a state pool and a streaming driver both need.
 *   <li>{@code ofConfined} - scratch inside one call on one thread. Cheaper close, no cross-thread
 *       handshake. Wrong for anything another thread touches: it fails loudly there.
 *   <li>{@code ofAuto} - READ_ONLY mapped weights, whose pages the kernel reclaims regardless, and
 *       bounded process-lifetime scratch with no owner to free it. NEVER for anonymous memory that
 *       scales with work: the heap stays small while native memory grows, so the GC that would
 *       free it never runs. That is not theoretical - it OOM-killed the integration battery at
 *       51 GB.
 *   <li>{@code global} - a one-model process: the CLI, the server.
 * </ul>
 *
 * <h2>The three laws the compiler cannot enforce</h2>
 *
 * <ol>
 *   <li>An arena must outlive every read from it. Kernels read raw addresses via {@code
 *       FloatTensor.GLOBAL_SEGMENT}, so the JDK's close handshake cannot save you: a live read
 *       from a closed arena is a CRASH, not an exception.
 *   <li>A weights arena must outlive every model sharing those weights.
 *   <li>Free on every path out, including the failing ones. A constructor that maps weights and
 *       then throws must close what it created before it rethrows - a leaked {@code ofShared}
 *       arena has no backstop.
 * </ol>
 *
 * <h2>What the code does enforce</h2>
 *
 * A state's {@code close()} is idempotent and BLOCKS until the in-flight computation returns, so
 * its returning is your quiescence certificate. Ownership is decided once, by the {@code newState}
 * flavour you called, and implemented once in this package - a family never decides it, which is
 * why a family cannot get it wrong. Every buffer of a state comes from that state's arena and no
 * other; {@code StateArenaDisciplineTest} pins it. With {@code -Djinfer.leakDetection}, a dropped
 * unclosed state names the line that created it.
 */
package com.qxotic.jinfer;

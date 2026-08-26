/**
 * Memory for tensors: storage ({@link com.qxotic.jota.memory.Memory}), shaped views over it ({@link
 * com.qxotic.jota.memory.MemoryView}) and the backend that owns it ({@link
 * com.qxotic.jota.memory.MemoryDomain}: allocator, element access, bulk operations).
 *
 * <h2>Constructors</h2>
 *
 * <p>{@link com.qxotic.jota.memory.Memories}, {@link com.qxotic.jota.memory.MemoryAllocators},
 * {@link com.qxotic.jota.memory.MemoryDomains} and {@link com.qxotic.jota.memory.MemoryViews} are
 * the entry points; the {@code internal} package is not API. Their method names state ownership:
 *
 * <ul>
 *   <li>{@code of(x)} wraps {@code x} and inherits its lifetime; nothing is allocated.
 *   <li>{@code adopt(x)} takes {@code x} over: closing the result closes {@code x}.
 *   <li>{@code newX()} creates something the caller owns; if it is {@link AutoCloseable}, the
 *       caller closes it.
 *   <li>a bare noun ({@code floats()}) is a shared, stateless instance; only the array backends
 *       qualify, because their storage is GC-managed.
 * </ul>
 *
 * <h2>Lifetimes</h2>
 *
 * <p>Native memory always names its lifetime; there is no default native allocator. {@link
 * com.qxotic.jota.memory.MemoryAllocators#ofArena(java.lang.foreign.Arena)} borrows the JDK arena
 * you pass ({@code ofConfined}, {@code ofShared}, {@code ofAuto}, {@code global}) and {@code
 * adoptArena} owns it: allocations are zero-filled, use after close throws {@link
 * IllegalStateException}, and only an adopted arena is closed through jota. {@link
 * com.qxotic.jota.memory.MemoryAllocators#newScopedArena()} is malloc-backed for frequent buffers:
 * each {@link com.qxotic.jota.memory.ScopedMemory} can be closed on its own, the arena closes the
 * rest; memory is not zero-filled and use after close is undefined. A {@link
 * com.qxotic.jota.memory.MemoryDomain} closes its allocator when closed; a borrowed arena ignores
 * that, an adopted or new one releases its memory.
 *
 * <p>{@link com.qxotic.jota.memory.Memory#isReadOnly()} is advisory: writers check it, but {@code
 * base()} hands out the storage and the JVM enforces nothing.
 *
 * <h2>Backends</h2>
 *
 * <p>Implement {@link com.qxotic.jota.memory.MemoryAllocator}, {@link
 * com.qxotic.jota.memory.MemoryOperations} and, where the host can address the memory, {@link
 * com.qxotic.jota.memory.MemoryAccess}, and bundle them in a {@code MemoryDomain}. Opaque backends
 * (GPUs) return {@code null} from {@code directAccess()}; strided copies and view constructors then
 * stage through host segments via {@code copyToNative}/{@code copyFromNative}, which accept heap
 * and native segments alike. Bounds and read-only checks go through {@link
 * com.qxotic.jota.memory.MemoryAccessChecks}; {@code -Djota.memory.checks=off|assert|runtime}
 * (default {@code runtime}) selects how they fail.
 *
 * <h2>Threads</h2>
 *
 * <p>Views and layouts are immutable. Reading a {@code Memory} concurrently is safe; writers
 * synchronize among themselves. Arenas are thread-safe for allocation; a confined JDK arena keeps
 * its own confinement rule.
 */
package com.qxotic.jota.memory;

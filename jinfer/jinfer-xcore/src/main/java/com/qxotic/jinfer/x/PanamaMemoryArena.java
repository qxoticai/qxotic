package com.qxotic.jinfer.x;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * A jota {@link MemoryArena} over a caller-supplied JDK {@link Arena} - an HONEST adapter, no
 * ownership policy of its own: {@link #close()} always delegates to the wrapped arena (freeing
 * every allocation, or throwing {@link UnsupportedOperationException} for self-managing arenas like
 * {@code ofAuto}/global). Whether close is ever CALLED - borrowed vs owned vs adopted - is the
 * holder's decision ({@code BaseState}); the same arena can be owned by one state and borrowed by
 * another, so ownership is a property of the relationship, never of this object. (The non-closeable
 * case is not exotic: in a native image {@code boundary.Arenas} degrades every arena to {@code
 * ofAuto}, so UOE-on-close is the NORMAL path there.)
 *
 * <p>jota's own native arenas ({@code NativeMemoryFactory.createArena/createManagedArena}) own
 * their lifecycle; jinfer's arenas are supplied at the boundary ({@code newState(..., Arena)},
 * runtime-adaptive shared/auto per {@code boundary.Arenas}), so this wraps rather than creates.
 * (ponytail: a candidate to push up to jota's nativeimpl if a second consumer appears.)
 */
public record PanamaMemoryArena(Arena arena) implements MemoryArena<MemorySegment> {

    public PanamaMemoryArena {
        Objects.requireNonNull(arena, "arena");
    }

    @Override
    public Device device() {
        return DeviceType.PANAMA.deviceIndex(0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    /** Kernels want cacheline/vector alignment; callers may still override per allocation. */
    @Override
    public long defaultByteAlignment() {
        return 64;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        return MemoryFactory.ofMemorySegment(arena.allocate(byteSize, byteAlignment));
    }

    @Override
    public void close() {
        arena.close();
    }

    /** The liveness canary {@link MemoryArena} doesn't expose; {@code BaseState.enter} runs it. */
    public boolean isAlive() {
        return arena.scope().isAlive();
    }
}

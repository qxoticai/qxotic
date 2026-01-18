package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.ScopedMemory;
import ai.qxotic.jota.memory.ScopedMemoryAllocatorArena;
import java.lang.foreign.MemorySegment;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

class UnsafeAllocatorArena implements ScopedMemoryAllocatorArena<MemorySegment> {

    private final Set<ScopedMemory<MemorySegment>> allocations =
            Collections.newSetFromMap(new ConcurrentHashMap<>());

    private UnsafeAllocatorArena() {}

    static ScopedMemoryAllocatorArena<MemorySegment> create() {
        return new UnsafeAllocatorArena();
    }

    @Override
    public Device device() {
        return Device.NATIVE;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public ScopedMemory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        ScopedMemory<MemorySegment> scopedMemory =
                UnsafeAllocator.instance().allocateMemory(byteSize, byteAlignment);
        allocations.add(scopedMemory);
        // TODO(peterssen): Avoid extra indirection, copy UnsafeAllocator and include memory
        // tracking.
        return new ScopedMemory<>() {
            @Override
            public void close() {
                allocations.remove(scopedMemory);
                scopedMemory.close();
            }

            @Override
            public long byteSize() {
                return scopedMemory.byteSize();
            }

            @Override
            public boolean isReadOnly() {
                return scopedMemory.isReadOnly();
            }

            @Override
            public Device device() {
                return scopedMemory.device();
            }

            @Override
            public MemorySegment base() {
                return scopedMemory.base();
            }

            @Override
            public long memoryGranularity() {
                return scopedMemory.memoryGranularity();
            }
        };
    }

    @Override
    public void close() {
        this.allocations.forEach(ScopedMemory::close);
        this.allocations.clear();
    }
}

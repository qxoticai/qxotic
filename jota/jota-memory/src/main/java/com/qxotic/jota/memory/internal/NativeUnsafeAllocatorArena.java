package com.qxotic.jota.memory.internal;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.ScopedArena;
import com.qxotic.jota.memory.ScopedMemory;
import java.lang.foreign.MemorySegment;
import java.util.HashSet;
import java.util.Set;

class NativeUnsafeAllocatorArena implements ScopedArena<MemorySegment> {

    private final Set<ScopedMemory<MemorySegment>> allocations = new HashSet<>();

    private volatile boolean closed;

    private NativeUnsafeAllocatorArena() {}

    static ScopedArena<MemorySegment> create() {
        return new NativeUnsafeAllocatorArena();
    }

    @Override
    public Device device() {
        return DeviceType.PANAMA.deviceIndex(0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public synchronized ScopedMemory<MemorySegment> allocateMemory(
            long byteSize, long byteAlignment) {
        if (closed) {
            throw new IllegalStateException("arena already closed");
        }
        ScopedMemory<MemorySegment> scopedMemory =
                NativeUnsafeAllocator.instance().allocateMemory(byteSize, byteAlignment);
        allocations.add(scopedMemory);
        return new ScopedMemory<>() {
            @Override
            public void close() {
                closeAllocation(scopedMemory);
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

            @Override
            public String toString() {
                long address = scopedMemory.base().address();
                StringBuilder sb = new StringBuilder("ArenaScopedMemory{address=0x");
                sb.append(Long.toHexString(address));
                sb.append(", byteSize=").append(byteSize());
                sb.append(", tracked=").append(isTracked(scopedMemory));
                if (isReadOnly()) {
                    sb.append(", readOnly=true");
                }
                sb.append('}');
                return sb.toString();
            }
        };
    }

    @Override
    public synchronized void close() {
        closed = true;
        allocations.forEach(ScopedMemory::close);
        allocations.clear();
    }

    private synchronized void closeAllocation(ScopedMemory<MemorySegment> allocation) {
        if (!allocations.remove(allocation)) {
            throw new IllegalStateException("memory already closed");
        }
        allocation.close();
    }

    private synchronized boolean isTracked(ScopedMemory<MemorySegment> allocation) {
        return allocations.contains(allocation);
    }

    /** {@code close()} frees every tracked allocation: dead afterwards. */
    @Override
    public boolean isAlive() {
        return !closed;
    }
}

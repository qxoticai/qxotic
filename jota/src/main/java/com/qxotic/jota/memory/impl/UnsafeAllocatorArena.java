package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.memory.ScopedMemory;
import com.qxotic.jota.memory.ScopedMemoryAllocatorArena;

import java.lang.foreign.MemorySegment;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

class UnsafeAllocatorArena implements ScopedMemoryAllocatorArena<MemorySegment> {

    private final Set<ScopedMemory<MemorySegment>> allocations = Collections.newSetFromMap(new ConcurrentHashMap<>());

    private UnsafeAllocatorArena() {
    }

    static ScopedMemoryAllocatorArena<MemorySegment> create() {
        return new UnsafeAllocatorArena();
    }

    @Override
    public Device device() {
        return Device.NATIVE;
    }

    @Override
    public ScopedMemory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        ScopedMemory<MemorySegment> scopedMemory = UnsafeAllocator.instance().allocateMemory(byteSize, byteAlignment);
        allocations.add(scopedMemory);
        // TODO(peterssen): Avoid extra indirection, copy UnsafeAllocator and include memory tracking.
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
            public boolean supportsDataType(DataType dataType) {
                return scopedMemory.supportsDataType(dataType);
            }
        };
    }

    @Override
    public void close() {
        this.allocations.forEach(ScopedMemory::close);
        this.allocations.clear();
    }
}

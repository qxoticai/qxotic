package com.qxotic.jota.memory.internal;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import java.util.Objects;

record ArrayMemoryDomain<B>(
        Class<B> arrayType,
        MemoryAllocator<B> memoryAllocator,
        MemoryAccess<B> directAccess,
        MemoryOperations<B> memoryOperations)
        implements MemoryDomain<B> {

    ArrayMemoryDomain {
        Objects.requireNonNull(arrayType);
        Objects.requireNonNull(memoryAllocator);
        Objects.requireNonNull(directAccess);
        Objects.requireNonNull(memoryOperations);
    }

    @Override
    public Device device() {
        return memoryAllocator.device();
    }

    @Override
    public void close() {
        // Arrays are managed by the GC.
    }

    @Override
    public String toString() {
        return "MemoryDomain{"
                + arrayType.getSimpleName()
                + ", device="
                + device()
                + ", directAccess=true}";
    }
}

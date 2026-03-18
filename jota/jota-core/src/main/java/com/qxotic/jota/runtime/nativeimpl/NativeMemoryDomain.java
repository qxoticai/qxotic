package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.*;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class NativeMemoryDomain implements MemoryDomain<MemorySegment> {

    private final MemoryAllocator<MemorySegment> memoryAllocator;

    NativeMemoryDomain(MemoryAllocator<MemorySegment> memoryAllocator) {
        assert memoryAllocator.device().belongsTo(DeviceType.PANAMA)
                : "Expected panama device, got " + memoryAllocator.device();
        this.memoryAllocator = Objects.requireNonNull(memoryAllocator);
    }

    @Override
    public Device device() {
        return memoryAllocator.device();
    }

    @Override
    public MemoryAllocator<MemorySegment> memoryAllocator() {
        return memoryAllocator;
    }

    @Override
    public MemoryAccess<MemorySegment> directAccess() {
        return NativeMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return NativeMemoryOperations.instance();
    }

    @Override
    public void close() {
        if (memoryAllocator instanceof MemoryArena<MemorySegment> memoryArena) {
            memoryArena.close();
        }
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{MemorySegment, device=")
                .append(device())
                .append(", directAccess=")
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}

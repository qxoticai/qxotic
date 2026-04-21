package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

record NativeMemoryDomain(MemoryAllocator<MemorySegment> memoryAllocator)
        implements MemoryDomain<MemorySegment> {

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
        return "Context{MemorySegment, device="
                + device()
                + ", directAccess="
                + (directAccess() != null)
                + '}';
    }
}

package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.*;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class PanamaContext implements MemoryContext<MemorySegment> {

    private final MemoryAllocator<MemorySegment> memoryAllocator;

    PanamaContext(MemoryAllocator<MemorySegment> memoryAllocator) {
        assert memoryAllocator.device().belongsTo(Device.PANAMA);
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
    public MemoryAccess<MemorySegment> memoryAccess() {
        return PanamaMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return PanamaMemoryOperations.instance();
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
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}

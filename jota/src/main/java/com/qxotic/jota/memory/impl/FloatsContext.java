package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.*;

class FloatsContext implements Context<float[]> {

    private static final FloatOperations<float[]> FLOAT_OPERATIONS = new GenericFloatOperations<>(FloatArrayMemoryAccess.instance());

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public MemoryAllocator<float[]> memoryAllocator() {
        return FloatArrayAllocator.instance();
    }

    @Override
    public MemoryAccess<float[]> memoryAccess() {
        return FloatArrayMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<float[]> memoryOperations() {
        return FloatsMemoryOperations.instance();
    }

    @Override
    public FloatOperations<float[]> floatOperations() {
        return FLOAT_OPERATIONS;
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return "context float[]";
    }
}

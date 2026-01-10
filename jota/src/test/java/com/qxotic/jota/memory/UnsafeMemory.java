package com.qxotic.jota.memory;

import com.qxotic.jota.Device;

public final class UnsafeMemory implements Memory<Void> {

    private static final Memory<Void> INSTANCE = new UnsafeMemory();

    public static Memory<Void> instance() {
        return INSTANCE;
    }

    private UnsafeMemory() {
    }

    @Override
    public long byteSize() {
        return Long.MAX_VALUE;
    }

    @Override
    public boolean isReadOnly() {
        return false;
    }

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public Void base() {
        return null;
    }
}

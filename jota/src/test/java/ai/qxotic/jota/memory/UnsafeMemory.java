package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;

public final class UnsafeMemory implements Memory<Void> {

    private static final Memory<Void> INSTANCE = new UnsafeMemory();

    public static Memory<Void> instance() {
        return INSTANCE;
    }

    private UnsafeMemory() {}

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
        return Device.PANAMA;
    }

    @Override
    public Void base() {
        return null;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return true;
    }

    @Override
    public String toString() {
        return new StringBuilder("Memory{unsafe, byteSize=")
                .append(byteSize())
                .append(", device=")
                .append(device())
                .append('}')
                .toString();
    }
}

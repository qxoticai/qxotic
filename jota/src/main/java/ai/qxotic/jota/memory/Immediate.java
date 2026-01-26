package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;

public record Immediate(long rawBits, DataType dataType) implements Storage {
    @Override
    public boolean isReadOnly() {
        return true;
    }
}

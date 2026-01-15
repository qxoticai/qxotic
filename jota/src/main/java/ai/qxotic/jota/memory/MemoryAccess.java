package ai.qxotic.jota.memory;

public interface MemoryAccess<B> {
    byte readByte(Memory<B> memory, long byteOffset);

    short readShort(Memory<B> memory, long byteOffset);

    int readInt(Memory<B> memory, long byteOffset);

    default float readFloat(Memory<B> memory, long byteOffset) {
        return Float.intBitsToFloat(readInt(memory, byteOffset));
    }

    long readLong(Memory<B> memory, long byteOffset);

    default double readDouble(Memory<B> memory, long byteOffset) {
        return Double.longBitsToDouble(readLong(memory, byteOffset));
    }

    void writeByte(Memory<B> memory, long byteOffset, byte value);

    void writeShort(Memory<B> memory, long byteOffset, short value);

    void writeInt(Memory<B> memory, long byteOffset, int value);

    default void writeFloat(Memory<B> memory, long byteOffset, float value) {
        writeInt(memory, byteOffset, Float.floatToRawIntBits(value));
    }

    void writeLong(Memory<B> memory, long byteOffset, long value);

    default void writeDouble(Memory<B> memory, long byteOffset, double value) {
        writeLong(memory, byteOffset, Double.doubleToRawLongBits(value));
    }
}

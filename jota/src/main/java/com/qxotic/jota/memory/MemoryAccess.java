package com.qxotic.jota.memory;

public interface MemoryAccess<B> {
    byte readByte(Memory<B> memory, long byteOffset);

    short readShort(Memory<B> memory, long byteOffset);

    int readInt(Memory<B> memory, long byteOffset);

    float readFloat(Memory<B> memory, long byteOffset);

    long readLong(Memory<B> memory, long byteOffset);

    double readDouble(Memory<B> memory, long byteOffset);

    void writeByte(Memory<B> memory, long byteOffset, byte value);

    void writeShort(Memory<B> memory, long byteOffset, short value);

    void writeInt(Memory<B> memory, long byteOffset, int value);

    void writeFloat(Memory<B> memory, long byteOffset, float value);

    void writeLong(Memory<B> memory, long byteOffset, long value);

    void writeDouble(Memory<B> memory, long byteOffset, double value);
}

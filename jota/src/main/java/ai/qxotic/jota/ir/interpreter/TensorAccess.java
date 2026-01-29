package ai.qxotic.jota.ir.interpreter;

import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;

final class TensorAccess<B> {

    private final MemoryAccess<B> access;

    TensorAccess(MemoryAccess<B> access) {
        this.access = access;
    }

    float readFloat(MemoryView<B> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readFloat(view.memory(), offset);
    }

    double readDouble(MemoryView<B> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readDouble(view.memory(), offset);
    }

    long readLong(MemoryView<B> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readLong(view.memory(), offset);
    }

    int readInt(MemoryView<B> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readInt(view.memory(), offset);
    }

    short readShort(MemoryView<B> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readShort(view.memory(), offset);
    }

    byte readByte(MemoryView<B> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readByte(view.memory(), offset);
    }

    void writeFloat(MemoryView<B> view, long linearIndex, float value) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        access.writeFloat(view.memory(), offset, value);
    }

    void writeDouble(MemoryView<B> view, long linearIndex, double value) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        access.writeDouble(view.memory(), offset, value);
    }

    void writeLong(MemoryView<B> view, long linearIndex, long value) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        access.writeLong(view.memory(), offset, value);
    }

    void writeInt(MemoryView<B> view, long linearIndex, int value) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        access.writeInt(view.memory(), offset, value);
    }

    void writeShort(MemoryView<B> view, long linearIndex, short value) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        access.writeShort(view.memory(), offset, value);
    }

    void writeByte(MemoryView<B> view, long linearIndex, byte value) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        access.writeByte(view.memory(), offset, value);
    }

    MemoryAccess<B> getAccess() {
        return access;
    }
}

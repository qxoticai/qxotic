package ai.llm4j.jota.memory;

import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;

public interface FloatValueIterator {

    boolean hasNext();

    float nextFloat();

    long elementCount();

    default void reset() {
        throw new UnsupportedOperationException();
    }
}

class FloatValueIteratorImpl implements FloatValueIterator {
    private final MemoryAccess<?> memoryAccess;
    private final OffsetIterator iterator;
    private final MemoryView<?> memoryView;

    <B> FloatValueIteratorImpl(MemoryView<B> memoryView, MemoryAccess<B> memoryAccess, OffsetIterator iterator) {
        this.memoryView = memoryView;
        this.memoryAccess = memoryAccess;
        this.iterator = iterator;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public float nextFloat() {
        throw new UnsupportedOperationException();
//        Memory<?> memory = memoryView.memory();
//        return memoryAccess.readFloat(memory, iterator.nextByteOffset());
    }

    @Override
    public long elementCount() {
        return 1;
    }

    @Override
    public void reset() {
        iterator.reset();
    }
}

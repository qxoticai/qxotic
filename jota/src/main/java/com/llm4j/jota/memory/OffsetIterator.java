package com.llm4j.jota.memory;

import com.llm4j.jota.memory.impl.ContiguousMemoryOffsetIterator;
import com.llm4j.jota.memory.impl.StridedOffsetIteratorImpl;
import com.qxotic.jota.memory.MemoryView;

public interface OffsetIterator {

    boolean hasNext();

    long nextByteOffset();

    long elementsInGroup();

    default void reset() {
        throw new UnsupportedOperationException();
    }

    static OffsetIterator create(MemoryView<?> memoryView) {
        if (memoryView.isContiguous()) {
            return new ContiguousMemoryOffsetIterator(memoryView);
        }
        return new StridedOffsetIteratorImpl(memoryView);
    }
}


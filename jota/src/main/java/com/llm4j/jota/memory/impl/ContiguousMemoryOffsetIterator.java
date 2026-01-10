package com.llm4j.jota.memory.impl;

import com.qxotic.jota.memory.MemoryView;
import com.llm4j.jota.memory.OffsetIterator;

public final class ContiguousMemoryOffsetIterator implements OffsetIterator {

    final long endOffset;
    final long stepSize;
    long byteOffset;

    public ContiguousMemoryOffsetIterator(MemoryView<?> memoryView) {
        if (!memoryView.isContiguous()) {
            throw new IllegalArgumentException("not contiguous view");
        }
        long totalNumberOfElements = memoryView.shape().size();
        this.stepSize = memoryView.dataType().byteSize();
        this.byteOffset = memoryView.byteOffset();
        this.endOffset = this.byteOffset + stepSize * totalNumberOfElements;
    }

    @Override
    public long nextByteOffset() {
        assert hasNext(); // assert for performance
        long currentByteOffset = byteOffset;
        byteOffset += stepSize;
        return currentByteOffset;
    }

    @Override
    public long elementsInGroup() {
        return 1;
    }

    @Override
    public boolean hasNext() {
        return byteOffset < endOffset;
    }
}

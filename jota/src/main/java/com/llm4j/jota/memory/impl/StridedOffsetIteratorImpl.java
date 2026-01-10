package com.llm4j.jota.memory.impl;

import com.llm4j.jota.memory.OffsetIterator;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.MemoryView;

public final class StridedOffsetIteratorImpl implements OffsetIterator {
    private final Stride byteStride;
    private final Layout layout;
    private final long startOffset;
    private final long totalElements;
    private final long[] currentPos;
    private long currentOffset;
    private long elementsProcessed;

    public StridedOffsetIteratorImpl(MemoryView<?> memoryView) {
        this.byteStride = memoryView.byteStride();
        this.layout = memoryView.layout();
        this.startOffset = memoryView.byteOffset();
        this.totalElements = memoryView.shape().size();
        this.currentPos = new long[layout.shape().rank()];
        this.currentOffset = startOffset;
        this.elementsProcessed = 0;
    }

    @Override
    public boolean hasNext() {
        return elementsProcessed < totalElements;
    }

    @Override
    public long nextByteOffset() {
        // assert for performance reasons, but should rather throw NoSuchElementException
        assert hasNext();

        long offsetToReturn = currentOffset;
        elementsProcessed++;

        // Increment the rightmost dimension
        int dim = layout.shape().rank() - 1;
        currentPos[dim]++;
        currentOffset += byteStride.flatAt(dim);

        // Handle dimension carry-over
        while (dim > 0 && currentPos[dim] >= layout.shape().flatAt(dim)) {
            currentPos[dim] = 0;
            dim--;
            currentPos[dim]++;
            currentOffset += byteStride.flatAt(dim) - (layout.shape().flatAt(dim + 1) * byteStride.flatAt(dim + 1));
        }

        return offsetToReturn;
    }

    @Override
    public long elementsInGroup() {
        return 1;
    }
}

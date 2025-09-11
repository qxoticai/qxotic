package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.MemoryView;
import com.llm4j.jota.memory.OffsetIterator;

public final class StridedOffsetIteratorImpl implements OffsetIterator {
    private final long[] byteStrides;
    private final Shape shape;
    private final long startOffset;
    private final long totalElements;
    private final long[] currentPos;
    private long currentOffset;
    private long elementsProcessed;

    public StridedOffsetIteratorImpl(MemoryView<?> memoryView) {
        this.byteStrides = memoryView.byteStrides();
        this.shape = memoryView.shape();
        this.startOffset = memoryView.byteOffset();
        this.totalElements = memoryView.shape().totalNumberOfElements();
        this.currentPos = new long[shape.rank()];
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
        int dim = shape.rank() - 1;
        currentPos[dim]++;
        currentOffset += byteStrides[dim];

        // Handle dimension carry-over
        while (dim > 0 && currentPos[dim] >= shape.dimension(dim)) {
            currentPos[dim] = 0;
            dim--;
            currentPos[dim]++;
            currentOffset += byteStrides[dim] - (shape.dimension(dim + 1) * byteStrides[dim + 1]);
        }

        return offsetToReturn;
    }

    @Override
    public long elementsInGroup() {
        return 1;
    }
}

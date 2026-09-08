package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/** State-owned scratch that grows to the largest synthesis seen, then reuses its buffers. */
final class KokoroWorkspace implements MemoryAllocator<MemorySegment> {

    interface Scope extends AutoCloseable {
        @Override
        void close();
    }

    private static final Scope NO_SCOPE = () -> {};

    private final MemoryAllocator<MemorySegment> arena;

    @SuppressWarnings("unchecked")
    private Memory<MemorySegment>[] buffers = new Memory[256];

    private long[] alignments = new long[buffers.length];
    private float[][] floatBuffers = new float[64][];
    private int[][] intBuffers = new int[16][];
    private float[][][] matrices = new float[16][][];
    private final StackScope[] scopes = new StackScope[16];
    private int top, floatTop, intTop, matrixTop, depth, backingAllocations;

    KokoroWorkspace(MemoryAllocator<MemorySegment> arena) {
        this.arena = arena;
        for (int i = 0; i < scopes.length; i++) scopes[i] = new StackScope();
    }

    void rewind() {
        top = 0;
        floatTop = 0;
        intTop = 0;
        matrixTop = 0;
        depth = 0;
    }

    private final class StackScope implements Scope {
        private int buffers, floats, ints, matrices;

        @Override
        public void close() {
            top = buffers;
            floatTop = floats;
            intTop = ints;
            matrixTop = matrices;
            depth--;
        }
    }

    private Scope scope() {
        StackScope scope = scopes[depth++];
        scope.buffers = top;
        scope.floats = floatTop;
        scope.ints = intTop;
        scope.matrices = matrixTop;
        return scope;
    }

    float[] floats(int size) {
        if (floatTop == floatBuffers.length)
            floatBuffers = Arrays.copyOf(floatBuffers, floatBuffers.length * 2);
        float[] buffer = floatBuffers[floatTop];
        if (buffer == null || buffer.length != size)
            buffer = floatBuffers[floatTop] = new float[size];
        floatTop++;
        return buffer;
    }

    int[] ints(int size) {
        if (intTop == intBuffers.length)
            intBuffers = Arrays.copyOf(intBuffers, intBuffers.length * 2);
        int[] buffer = intBuffers[intTop];
        if (buffer == null || buffer.length != size) buffer = intBuffers[intTop] = new int[size];
        intTop++;
        return buffer;
    }

    float[][] matrix(int rows, int columns) {
        if (matrixTop == matrices.length) matrices = Arrays.copyOf(matrices, matrices.length * 2);
        float[][] matrix = matrices[matrixTop];
        if (matrix == null || matrix.length != rows || (rows != 0 && matrix[0].length != columns)) {
            matrix = matrices[matrixTop] = new float[rows][columns];
        }
        matrixTop++;
        return matrix;
    }

    static float[] takeFloats(MemoryAllocator<MemorySegment> allocator, int size) {
        return allocator instanceof KokoroWorkspace workspace
                ? workspace.floats(size)
                : new float[size];
    }

    static int[] takeInts(MemoryAllocator<MemorySegment> allocator, int size) {
        return allocator instanceof KokoroWorkspace workspace
                ? workspace.ints(size)
                : new int[size];
    }

    static float[][] takeMatrix(MemoryAllocator<MemorySegment> allocator, int rows, int columns) {
        return allocator instanceof KokoroWorkspace workspace
                ? workspace.matrix(rows, columns)
                : new float[rows][columns];
    }

    static Scope scope(MemoryAllocator<MemorySegment> allocator) {
        return allocator instanceof KokoroWorkspace workspace ? workspace.scope() : NO_SCOPE;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (top == buffers.length) {
            buffers = Arrays.copyOf(buffers, buffers.length * 2);
            alignments = Arrays.copyOf(alignments, alignments.length * 2);
        }
        Memory<MemorySegment> buffer = buffers[top];
        if (buffer == null || buffer.byteSize() < byteSize || alignments[top] < byteAlignment) {
            buffer = buffers[top] = arena.allocateMemory(byteSize, byteAlignment);
            alignments[top] = byteAlignment;
            backingAllocations++;
        }
        top++;
        return buffer;
    }

    int backingAllocations() {
        return backingAllocations;
    }

    @Override
    public Device device() {
        return arena.device();
    }

    @Override
    public long defaultByteAlignment() {
        return arena.defaultByteAlignment();
    }

    @Override
    public long memoryGranularity() {
        return arena.memoryGranularity();
    }
}

package com.qxotic.jinfer;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * State-owned scratch that grows to the largest request seen, then reuses its buffers: a model
 * state holds one and {@link #rewind}s it per call, so steady-state inference allocates nothing.
 * Buffers are handed out by allocation order, so a call must allocate in the same order each time
 * to reuse them, and they come back holding the previous call's data - never zeroed.
 */
public final class Workspace implements MemoryAllocator<MemorySegment> {

    public interface Scope extends AutoCloseable {
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
    private int top, floatTop, intTop, matrixTop, depth, backingAllocations, heapAllocations;

    public Workspace(MemoryAllocator<MemorySegment> arena) {
        this.arena = arena;
        for (int i = 0; i < scopes.length; i++) scopes[i] = new StackScope();
    }

    public void rewind() {
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

    public float[] floats(int size) {
        if (floatTop == floatBuffers.length)
            floatBuffers = Arrays.copyOf(floatBuffers, floatBuffers.length * 2);
        float[] buffer = floatBuffers[floatTop];
        if (buffer == null || buffer.length != size)
            buffer = floatBuffers[floatTop] = newFloats(size);
        floatTop++;
        return buffer;
    }

    /**
     * As {@link #floats}, but any pooled array of at least {@code size} is reused: for callers that
     * index explicitly (never through {@code length}) and see sizes vary call to call.
     */
    public float[] floatsAtLeast(int size) {
        if (floatTop == floatBuffers.length)
            floatBuffers = Arrays.copyOf(floatBuffers, floatBuffers.length * 2);
        float[] buffer = floatBuffers[floatTop];
        if (buffer == null || buffer.length < size)
            buffer = floatBuffers[floatTop] = newFloats(size);
        floatTop++;
        return buffer;
    }

    private float[] newFloats(int size) {
        heapAllocations++;
        return new float[size];
    }

    public int[] ints(int size) {
        if (intTop == intBuffers.length)
            intBuffers = Arrays.copyOf(intBuffers, intBuffers.length * 2);
        int[] buffer = intBuffers[intTop];
        if (buffer == null || buffer.length != size) {
            buffer = intBuffers[intTop] = new int[size];
            heapAllocations++;
        }
        intTop++;
        return buffer;
    }

    public float[][] matrix(int rows, int columns) {
        if (matrixTop == matrices.length) matrices = Arrays.copyOf(matrices, matrices.length * 2);
        float[][] matrix = matrices[matrixTop];
        if (matrix == null || matrix.length != rows || (rows != 0 && matrix[0].length != columns)) {
            matrix = matrices[matrixTop] = new float[rows][columns];
            heapAllocations++;
        }
        matrixTop++;
        return matrix;
    }

    public static float[] takeFloats(MemoryAllocator<MemorySegment> allocator, int size) {
        return allocator instanceof Workspace workspace ? workspace.floats(size) : new float[size];
    }

    public static int[] takeInts(MemoryAllocator<MemorySegment> allocator, int size) {
        return allocator instanceof Workspace workspace ? workspace.ints(size) : new int[size];
    }

    public static float[][] takeMatrix(
            MemoryAllocator<MemorySegment> allocator, int rows, int columns) {
        return allocator instanceof Workspace workspace
                ? workspace.matrix(rows, columns)
                : new float[rows][columns];
    }

    public static Scope scope(MemoryAllocator<MemorySegment> allocator) {
        return allocator instanceof Workspace workspace ? workspace.scope() : NO_SCOPE;
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

    /** Native buffers allocated so far: flat across calls once the workspace is warm. */
    public int backingAllocations() {
        return backingAllocations;
    }

    /** Heap arrays allocated so far: flat across calls once the workspace is warm. */
    public int heapAllocations() {
        return heapAllocations;
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

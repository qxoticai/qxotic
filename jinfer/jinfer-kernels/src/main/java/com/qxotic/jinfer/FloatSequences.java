package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Bridges between the {@link FloatSequence} potato world (core API) and the {@link FloatTensor}
 * kernel world (this module). Producers hand out {@link #view} potatoes over their tensor buffers;
 * consumers crack them back with {@link #asTensor} - zero-copy for the backings this module knows
 * (native tensors, heap arrays), one honest copy for anything exotic.
 */
public final class FloatSequences {

    private FloatSequences() {}

    /** A potato over the whole tensor - read-only face, {@code spi.Native} hatch inside. */
    public static FloatSequence view(FloatTensor tensor) {
        return new NativeFloatSequence(tensor, 0, Math.toIntExact(tensor.size()));
    }

    /** A potato over {@code tensor[offset .. offset+length)}. */
    public static FloatSequence view(FloatTensor tensor, long offset, int length) {
        return new NativeFloatSequence(tensor, offset, length);
    }

    /**
     * The inverse bridge, for kernel consumers (a forward ingesting {@code
     * Batch.Input.Embeddings}): the backing tensor itself for a whole-tensor {@code Native} potato;
     * a zero-copy heap-array tensor for a {@code Heap} one; ONE copy into a fresh heap tensor for
     * anything else. The result is a READ-SIDE bridge (gather/copyTo/scalar ops) - it is not a gemm
     * operand.
     */
    public static FloatTensor asTensor(FloatSequence seq) {
        if (seq instanceof com.qxotic.jinfer.spi.Native n
                && n.offset() == 0
                && seq.size() == n.tensor().size()) {
            return n.tensor();
        }
        if (seq instanceof com.qxotic.jinfer.spi.Heap h) {
            return new HeapF32FloatTensor(h.array(), h.offset(), seq.size());
        }
        float[] a = seq.toArray();
        return new HeapF32FloatTensor(a, 0, a.length);
    }
}

/** The native-tensor potato: read-only {@link FloatSequence} face, {@code spi.Native} hatch. */
final class NativeFloatSequence implements FloatSequence, com.qxotic.jinfer.spi.Native {

    private final FloatTensor tensor;
    private final long offset;
    private final int length;

    NativeFloatSequence(FloatTensor tensor, long offset, int length) {
        if (offset < 0 || length < 0 || offset + length > tensor.size())
            throw new IndexOutOfBoundsException(offset + " + " + length + " > " + tensor.size());
        this.tensor = tensor;
        this.offset = offset;
        this.length = length;
    }

    @Override
    public int size() {
        return length;
    }

    @Override
    public float get(int index) {
        if (index < 0 || index >= length) throw new IndexOutOfBoundsException(index);
        return tensor.getFloat(offset + index);
    }

    @Override
    public void copyTo(float[] dst, int start, int count) {
        if (start < 0 || count < 0 || start + count > length || count > dst.length)
            throw new IndexOutOfBoundsException(start + " + " + count + " / " + length);
        tensor.copyRow(offset + start, dst, 0, count);
    }

    @Override
    public FloatTensor tensor() {
        return tensor;
    }

    @Override
    public long offset() {
        return offset;
    }
}

/**
 * A heap-array-backed F32 tensor - the zero-copy landing for {@code Heap} potatoes into kernel
 * consumers. READ-SIDE bridge only: scalar/gather/copy ops are plain array accesses, but there is
 * no vector view ({@link #getFloatVector} throws), so it must never be a gemm operand.
 */
final class HeapF32FloatTensor extends FloatTensor {

    private final float[] array;
    private final int offset;
    private final int length;

    HeapF32FloatTensor(float[] array, int offset, int length) {
        this.array = array;
        this.offset = offset;
        this.length = length;
    }

    @Override
    public long size() {
        return length;
    }

    @Override
    public float getFloat(long index) {
        return array[offset + Math.toIntExact(index)];
    }

    @Override
    public void setFloat(long index, float value) {
        array[offset + Math.toIntExact(index)] = value;
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long offset) {
        throw new UnsupportedOperationException("getFloatVector: heap tensor is a read bridge");
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public void copyRow(long srcOff, float[] dst, int dstOff, int count) {
        System.arraycopy(array, offset + Math.toIntExact(srcOff), dst, dstOff, count);
    }
}

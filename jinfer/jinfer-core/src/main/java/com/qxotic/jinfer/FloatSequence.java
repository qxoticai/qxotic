package com.qxotic.jinfer;

/**
 * A read-only handle on a run of floats - a hot potato, not an owning value. Pass it, read it,
 * don't store it: validity is the producer's contract (for {@code LanguageModel.logits}: until the
 * next {@code logits}/{@code ingest} on that state; for an {@code Embedder} chunk: until the next
 * sink call). To own the data, copy it out ({@link #toArray()} / {@link #copyTo}).
 *
 * <p>The interface never says what's inside - a heap array, native memory, something on a GPU.
 * Library internals that know how to drive a particular backing crack the potato via the SPI
 * hatches ({@code com.qxotic.jinfer.spi.Heap} in this module, kernels-side hatches in
 * jinfer-kernels); {@link #copyTo} is the universal fallback that makes every potato readable
 * everywhere. Implementations MUST NOT hand out mutation through the public face.
 */
public interface FloatSequence {

    int size();

    float get(int index);

    /** Copies {@code this[start .. start+count)} into {@code dst[0 .. count)}. */
    void copyTo(float[] dst, int start, int count);

    /** An owning copy of the whole sequence. */
    default float[] toArray() {
        float[] dst = new float[size()];
        copyTo(dst, 0, size());
        return dst;
    }

    /** An OWNING copy of {@code values} - safe to retain, safe to share. */
    static FloatSequence of(float[] values) {
        return new FloatArray(values.clone(), 0, values.length);
    }

    /** Trusted zero-copy wrap of a whole array: the caller must not mutate or share it after. */
    static FloatSequence wrap(float[] values) {
        return new FloatArray(values, 0, values.length);
    }

    /** Trusted zero-copy wrap of {@code values[offset .. offset+length)} - same contract. */
    static FloatSequence wrap(float[] values, int offset, int length) {
        return new FloatArray(values, offset, length);
    }
}

/** The heap-backed potato, and the {@code spi.Heap} hatch the sampler/embedder internals crack. */
final class FloatArray implements FloatSequence, com.qxotic.jinfer.spi.Heap {

    private final float[] array;
    private final int offset;
    private final int length;

    FloatArray(float[] array, int offset, int length) {
        if (offset < 0 || length < 0 || offset + length > array.length)
            throw new IndexOutOfBoundsException(offset + " + " + length + " > " + array.length);
        this.array = array;
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
        return array[offset + index];
    }

    @Override
    public void copyTo(float[] dst, int start, int count) {
        if (start < 0 || count < 0 || start + count > length || count > dst.length)
            throw new IndexOutOfBoundsException(start + " + " + count + " / " + length);
        System.arraycopy(array, offset + start, dst, 0, count);
    }

    @Override
    public float[] array() {
        return array;
    }

    @Override
    public int offset() {
        return offset;
    }
}

package com.qxotic.model.llm.llama;

import com.qxotic.span.FloatSpan;
import java.util.stream.IntStream;

public final class ArraySpan implements FloatSpan {

    final float[] values;
    final int offset;
    final int size;

    ArraySpan(float[] values, int offset, int size) {
        Util.checkBounds(0 <= size && size <= values.length, "invalid size");
        // Util.checkArg(0 <= offset && offset <= size, "invalid offset");
        Util.checkBounds(offset <= values.length - size, "slice ouf of bounds");
        this.values = values;
        this.offset = offset;
        this.size = size;
    }

    public static ArraySpan allocate(int... dims) {
        assert dims.length > 0;
        assert IntStream.of(dims).allMatch(i -> i > 0);
        int arraySize = IntStream.of(dims).reduce(1, Math::multiplyExact);
        return new ArraySpan(new float[arraySize], 0, arraySize);
    }

    public static FloatSpan wrap(float[] values) {
        return new ArraySpan(values, 0, values.length);
    }

    @Override
    public long size() {
        return this.size;
    }

    @Override
    public long sizeInBytes() {
        return size() * Float.BYTES;
    }

    @Override
    public FloatSpan slice(long spanStartIndex, long sliceSize) {
        Util.checkBounds(0 <= spanStartIndex && spanStartIndex <= size, "invalid offset");
        Util.checkBounds(sliceSize <= size, "invalid size");
        Util.checkBounds(spanStartIndex <= size - sliceSize, "slice ouf of bounds");
        return new ArraySpan(this.values, this.offset + (int) spanStartIndex, (int) sliceSize);
    }
}

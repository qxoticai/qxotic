package com.qxotic.jam.internal;

import java.lang.foreign.MemorySegment;

/** Safety checks shared by JAM backends that access operands through raw native addresses. */
public final class MemoryChecks {

    private MemoryChecks() {}

    public static void requireNative(MemorySegment segment, String name) {
        if (!segment.scope().isAlive())
            throw new IllegalStateException("jam.mm: " + name + " segment is not alive");
        if (!segment.isNative())
            throw new IllegalArgumentException(
                    "jam.mm: "
                            + name
                            + " must be a NATIVE (off-heap) MemorySegment - heap/array-backed has"
                            + " no native address");
    }

    public static void checkSegment(
            String name,
            MemorySegment segment,
            long offset,
            int dtype,
            int stride,
            int rows,
            int rowElements) {
        GGMLType type = GGMLType.byCode(dtype);
        if (type == null) return;
        long required;
        try {
            required = type.spanBytes(rows, stride, rowElements);
        } catch (ArithmeticException e) {
            throw new IndexOutOfBoundsException("jam.mm: " + name + " byte span overflows");
        }
        if (offset < 0 || offset > segment.byteSize() - required)
            throw new IndexOutOfBoundsException(
                    "jam.mm: "
                            + name
                            + " segment too small - need "
                            + required
                            + " B at offset "
                            + offset
                            + ", segment is "
                            + segment.byteSize()
                            + " B");
    }
}

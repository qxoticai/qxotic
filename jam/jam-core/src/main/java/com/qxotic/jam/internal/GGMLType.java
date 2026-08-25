package com.qxotic.jam.internal;

import com.qxotic.jam.JAM;

/**
 * Internal block geometry shared by JAM's built-in providers. The public dtype surface remains the
 * {@code int} tags on {@link JAM}.
 *
 * <p>Only the dtypes jam runs are listed — a one-to-one mirror of {@link JAM}'s tags (a test keeps
 * them in sync). An unrecognized code -> {@link #byCode} returns {@code null}. Codes match
 * GGML/GGUF, but jam keeps its OWN copy here and carries <b>no dependency</b> on {@code
 * com.qxotic.gguf}.
 */
public enum GGMLType {
    //    ggml  blockElems  blockBytes
    F32(0, 1, 4),
    F16(1, 1, 2),
    BF16(30, 1, 2),
    Q4_0(2, 32, 18),
    Q8_0(8, 32, 34),
    Q4_K(12, 256, 144),
    Q5_K(13, 256, 176),
    Q6_K(14, 256, 210),
    MXFP4(39, 32, 17),
    NVFP4(40, 64, 36),
    Q1_0(41, 128, 18);

    private final int code;
    private final int elementsPerBlock;
    private final int bytesPerBlock;

    GGMLType(int code, int elementsPerBlock, int bytesPerBlock) {
        this.code = code;
        this.elementsPerBlock = elementsPerBlock;
        this.bytesPerBlock = bytesPerBlock;
    }

    public int code() {
        return code;
    }

    public int elementsPerBlock() {
        return elementsPerBlock;
    }

    public int bytesPerBlock() {
        return bytesPerBlock;
    }

    /** Byte span of {@code elements} consecutive elements of this dtype (block multiple). */
    public long rowBytes(long elements) {
        return elements / elementsPerBlock * (long) bytesPerBlock;
    }

    /**
     * Bytes the kernel touches for an operand of {@code rows} rows — {@code rowElems} data elements
     * each, at ELEMENT row-stride {@code stride}: {@code (rows-1)} full strides plus the last row's
     * data. This is the element-stride → byte-span conversion {@link JAM#mm}'s bounds check needs
     * (it then compares this to {@code MemorySegment.byteSize()}).
     */
    public long spanBytes(int rows, int stride, int rowElems) {
        return (long) (rows - 1) * rowBytes(stride) + rowBytes(rowElems);
    }

    /** O(1) code → dtype lookup; {@code null} for an unrecognized or unsupported code. */
    public static GGMLType byCode(int code) {
        return (code >= 0 && code < BY_CODE.length) ? BY_CODE[code] : null;
    }

    private static final GGMLType[] BY_CODE =
            new GGMLType[42]; // codes 0..41 (Q1_0); gaps stay null

    static {
        for (GGMLType type : values()) BY_CODE[type.code] = type;
    }
}

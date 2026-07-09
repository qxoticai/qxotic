package com.qxotic.jam;

/**
 * INTERNAL dtype geometry — elements-per-block + bytes-per-block of jam's supported weight dtypes.
 * NOT public API: package-private, but shared across the jam-* modules — they all live in the
 * {@code com.qxotic.jam} package, so same-package access works on the classpath (jam-native's
 * bounds check, jam-scalar/jam-vector's decode). The public dtype surface is the {@code int} tags
 * on {@link JAM} (mirroring GGML's {@code ggml_type}).
 *
 * <p>Only the dtypes jam runs are listed — a one-to-one mirror of {@link JAM}'s tags (a test keeps
 * them in sync). An unrecognized code -> {@link #byCode} returns {@code null}. Codes match
 * GGML/GGUF, but jam keeps its OWN copy here and carries <b>no dependency</b> on {@code
 * com.qxotic.gguf}.
 */
enum GGMLType {
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
    NVFP4(40, 64, 36);

    final int ggml; // ggml_type code (the GGUF on-disk tag)
    final int blockElems; // elements per quant block
    final int blockBytes; // bytes per quant block

    GGMLType(int ggml, int blockElems, int blockBytes) {
        this.ggml = ggml;
        this.blockElems = blockElems;
        this.blockBytes = blockBytes;
    }

    /** Byte span of {@code elements} consecutive elements of this dtype (block multiple). */
    long rowBytes(long elements) {
        return elements / blockElems * (long) blockBytes;
    }

    /**
     * Bytes the kernel touches for an operand of {@code rows} rows — {@code rowElems} data elements
     * each, at ELEMENT row-stride {@code stride}: {@code (rows-1)} full strides plus the last row's
     * data. This is the element-stride → byte-span conversion {@link JAM#mm}'s bounds check needs
     * (it then compares this to {@code MemorySegment.byteSize()}).
     */
    long spanBytes(int rows, int stride, int rowElems) {
        return (long) (rows - 1) * rowBytes(stride) + rowBytes(rowElems);
    }

    /** O(1) code → dtype lookup; {@code null} for an unrecognized or unsupported code. */
    static GGMLType byCode(int ggml) {
        return (ggml >= 0 && ggml < BY_CODE.length) ? BY_CODE[ggml] : null;
    }

    private static final GGMLType[] BY_CODE =
            new GGMLType[41]; // codes 0..40 (NVFP4); gaps stay null

    static {
        for (GGMLType q : values()) BY_CODE[q.ggml] = q;
    }
}

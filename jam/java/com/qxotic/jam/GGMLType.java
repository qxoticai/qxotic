package com.qxotic.jam;

/**
 * INTERNAL implementation convenience — package-private, NOT part of jam's public API. The public surface is
 * the {@code int} dtype tags on {@link JAM} (which mirror GGML's {@code ggml_type}); this enum just gathers
 * their block geometry in one place, used only by the Java-side bounds check in {@link JAM#mm}. It is never
 * exposed in a public signature.
 *
 * <p>Only the dtypes jam actually runs are listed — a one-to-one mirror of {@link JAM}'s public tags (a test
 * keeps them in sync). Any other code (an unsupported GGUF tag, or garbage) isn't here, so {@link #byCode}
 * returns {@code null} and the bounds check skips it: {@code jam_mm} then returns {@code EUNSUPPORTED}
 * without ever reading the weight. The codes are numerically identical to GGML / GGUF (hence
 * {@code com.qxotic.gguf}), but jam keeps its OWN copy here and carries <b>no dependency</b> on that module.
 */
enum GGMLType {
    //    ggml  blockElems  blockBytes
    F32  ( 0,     1,           4),
    F16  ( 1,     1,           2),
    BF16 (30,     1,           2),
    Q4_0 ( 2,    32,          18),
    Q8_0 ( 8,    32,          34),
    Q4_K (12,   256,         144),
    Q5_K (13,   256,         176),
    Q6_K (14,   256,         210),
    MXFP4(39,    32,          17),
    NVFP4(40,    64,          36);

    final int ggml;           // ggml_type code (the GGUF on-disk tag)
    final int blockElems;     // elements per quant block
    final int blockBytes;     // bytes per quant block

    GGMLType(int ggml, int blockElems, int blockBytes) {
        this.ggml = ggml;
        this.blockElems = blockElems;
        this.blockBytes = blockBytes;
    }

    /** Byte span of {@code elements} consecutive elements of this dtype (block multiple). */
    long rowBytes(long elements) { return elements / blockElems * (long) blockBytes; }

    /** Bytes the kernel touches for an operand of {@code rows} rows — {@code rowElems} data elements each, at
     *  ELEMENT row-stride {@code stride}: {@code (rows-1)} full strides plus the last row's data. This is the
     *  element-stride → byte-span conversion {@link JAM#mm}'s bounds check needs (it then compares this to
     *  {@code MemorySegment.byteSize()}). */
    long spanBytes(int rows, int stride, int rowElems) {
        return (long) (rows - 1) * rowBytes(stride) + rowBytes(rowElems);
    }

    /** O(1) code → dtype lookup; {@code null} for an unrecognized or unsupported code. */
    static GGMLType byCode(int ggml) { return (ggml >= 0 && ggml < BY_CODE.length) ? BY_CODE[ggml] : null; }

    private static final GGMLType[] BY_CODE = new GGMLType[41];   // codes 0..40 (NVFP4); gaps stay null
    static { for (GGMLType q : values()) BY_CODE[q.ggml] = q; }
}

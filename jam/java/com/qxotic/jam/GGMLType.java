package com.qxotic.jam;

/**
 * INTERNAL implementation convenience — package-private, NOT part of jam's public API. The public surface is
 * the {@code int} dtype tags on {@link JAM} (which mirror GGML's {@code ggml_type}); this enum just gathers
 * their block geometry and supported-set in one place, used only by the Java-side bounds check in
 * {@link JAM#mm}. It is never exposed in a public signature.
 *
 * <p>The codes are numerically identical to GGML / GGUF (hence {@code com.qxotic.gguf}), so a GGUF tag maps
 * straight through — but jam keeps its OWN copy here and carries <b>no dependency</b> on the GGUF module.
 * {@code supported} = jam has a kernel; the rest are recognized tags that {@code jam_mm} rejects with
 * {@code EUNSUPPORTED} without reading the weight (so the bounds check skips them). A test keeps these codes
 * in sync with {@link JAM}'s public int constants.
 */
enum GGMLType {
    //    ggml  blockElems  blockBytes  supported
    F32  ( 0,     1,           4,        true),
    F16  ( 1,     1,           2,        true),
    Q4_0 ( 2,    32,          18,        true),
    Q4_1 ( 3,    32,          20,        false),
    Q5_0 ( 6,    32,          22,        false),
    Q5_1 ( 7,    32,          24,        false),
    Q8_0 ( 8,    32,          34,        true),
    Q2_K (10,   256,          84,        false),
    Q3_K (11,   256,         110,        false),
    Q4_K (12,   256,         144,        true),
    Q5_K (13,   256,         176,        true),
    Q6_K (14,   256,         210,        true),
    Q8_K (15,   256,         292,        false),
    BF16 (30,     1,           2,        true),
    MXFP4(39,    32,          17,        true),
    NVFP4(40,    64,          36,        true);

    final int ggml;           // ggml_type code (the GGUF on-disk tag)
    final int blockElems;     // elements per quant block
    final int blockBytes;     // bytes per quant block
    final boolean supported;  // jam has a kernel (else jam_mm returns EUNSUPPORTED without reading the weight)

    GGMLType(int ggml, int blockElems, int blockBytes, boolean supported) {
        this.ggml = ggml;
        this.blockElems = blockElems;
        this.blockBytes = blockBytes;
        this.supported = supported;
    }

    /** Byte span of {@code elements} consecutive elements of this dtype (block multiple). */
    long rowBytes(long elements) { return elements / blockElems * (long) blockBytes; }

    /** Bytes the kernel touches for an operand of {@code rows} rows — {@code rowElems} data elements each, at
     *  ELEMENT row-stride {@code stride}: {@code (rows-1)} full strides plus the last row's data. This is the
     *  element-stride → byte-span conversion {@link JAM#mm}'s bounds check needs (it then compares this to
     *  {@code MemorySegment.byteSize()}). Only called for a {@link #supported} dtype. */
    long spanBytes(int rows, int stride, int rowElems) {
        return (long) (rows - 1) * rowBytes(stride) + rowBytes(rowElems);
    }

    /** O(1) code → dtype lookup; {@code null} for an unrecognized code. */
    static GGMLType byCode(int ggml) { return (ggml >= 0 && ggml < BY_CODE.length) ? BY_CODE[ggml] : null; }

    private static final GGMLType[] BY_CODE = new GGMLType[41];   // codes 0..40 (NVFP4); gaps stay null
    static { for (GGMLType q : values()) BY_CODE[q.ggml] = q; }
}

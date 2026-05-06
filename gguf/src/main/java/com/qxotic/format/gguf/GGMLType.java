package com.qxotic.format.gguf;

/**
 * GGML tensor data types.
 *
 * <p>Each type defines how tensor data is encoded in memory. Types fall into three categories:
 *
 * <ul>
 *   <li><b>Primitive types</b> - F32, F16, BF16, I8, I16, I32, I64, F64
 *   <li><b>Legacy quantization</b> - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
 *   <li><b>K-quantization</b> - Q2_K through Q8_K, IQ series, TQ series, MXFP4, NVFP4
 * </ul>
 *
 * <p>Quantized types store elements in blocks. Each block has:
 *
 * <ul>
 *   <li>A fixed number of elements ({@link #getElementsPerBlock()})
 *   <li>A fixed byte size ({@link #getBlockByteSize()})
 *   <li>Optional scale factors and metadata
 * </ul>
 *
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/include/ggml.h">GGML Header</a>
 */
public enum GGMLType {

    // ============================================================
    // Primitive Types (elements per block = 1)
    // ============================================================

    /** 32-bit IEEE 754 single-precision floating point. 32.0 bpw. */
    F32(4, 1),

    /** 16-bit IEEE 754 half-precision floating point (FP16). 16.0 bpw. */
    F16(2, 1),

    /**
     * Layout: [{d: fp16, qs: uint8[16]}]. 1 fp16 scale + 32 packed 4-bit weights (nibbles). 4.5
     * bpw.
     */
    Q4_0(18, 32),

    /**
     * Layout: [{d: fp16, m: fp16, qs: uint8[16]}]. 1 fp16 scale + 1 fp16 min + 32 packed 4-bit
     * weights. Weight = d * q + m. 5.0 bpw.
     */
    Q4_1(20, 32),

    /**
     * @deprecated Support has been removed from GGML.
     */
    @Deprecated
    Q4_2(0, 0),

    /**
     * @deprecated Support has been removed from GGML.
     */
    @Deprecated
    Q4_3(0, 0),

    /**
     * Layout: [{d: fp16, qh: uint32, qs: uint8[16]}]. 1 fp16 scale + 32 high bits packed as uint32
     * + 32 low 4-bit weights. 5.5 bpw.
     */
    Q5_0(22, 32),

    /**
     * Layout: [{d: fp16, m: fp16, qh: uint32, qs: uint8[16]}]. 1 fp16 scale + 1 fp16 min + 32 high
     * bits packed as uint32 + 32 low 4-bit weights. Weight = d * q + m. 6.0 bpw.
     */
    Q5_1(24, 32),

    /** Layout: [{d: fp16, qs: int8[32]}]. 1 fp16 scale + 32 signed 8-bit weights. 8.5 bpw. */
    Q8_0(34, 32),

    /**
     * Layout: [{d: fp16, s: fp16, qs: int8[32]}]. 1 fp16 scale + 1 fp16 sum (d * Σqs[i]) + 32
     * signed 8-bit weights. Used for dot product optimization. 9.0 bpw.
     */
    Q8_1(36, 32),

    // ============================================================
    // K-Quantization Types (elements per block = 256)
    // ============================================================

    /**
     * Layout: [{d: fp16, dmin: fp16, scales: uint8[16], qs: uint8[64]}]. 16 sub-blocks of 16. Per
     * sub-block: 4-bit scale + 4-bit min packed in scales[16]; 2-bit weights. Weight = d * scale[i]
     * * q + dmin * min[i]. 2.625 bpw.
     */
    Q2_K(84, 256),

    /**
     * Layout: [{hmask: uint8[32], qs: uint8[64], scales: uint8[12], d: fp16}]. 16 sub-blocks of 16.
     * hmask + qs = 3-bit weights (qs = low 2 bits). 12 bytes hold 6-bit scales for 16 sub-blocks.
     * Weight = d * scale[i] * q. 3.4375 bpw.
     */
    Q3_K(110, 256),

    /**
     * Layout: [{d: fp16, dmin: fp16, scales: uint8[12], qs: uint8[128]}]. 8 sub-blocks of 32. 12
     * bytes hold 6-bit scale + 6-bit min per sub-block; 4-bit weights. Weight = d * scale[i] * q +
     * dmin * min[i]. 4.5 bpw.
     */
    Q4_K(144, 256),

    /**
     * Layout: [{d: fp16, dmin: fp16, scales: uint8[12], qh: uint8[32], qs: uint8[128]}]. 8
     * sub-blocks of 32. 12 bytes for 6-bit scale + 6-bit min; qh has high bits, qs has low 4 bits =
     * 5-bit weights. Weight = d * scale[i] * q + dmin * min[i]. 5.5 bpw.
     */
    Q5_K(176, 256),

    /**
     * Layout: [{ql: uint8[128], qh: uint8[64], scales: int8[16], d: fp16}]. 16 sub-blocks of 16. ql
     * = low 4 bits, qh = high 2 bits = 6-bit weights. 1 int8 scale per sub-block. Weight = d *
     * scale[i] * q. 6.5625 bpw.
     */
    Q6_K(210, 256),

    /**
     * Layout: [{d: fp32, qs: int8[256], bsums: int16[16]}]. 16 sub-blocks of 16. fp32 scale + 256
     * signed 8-bit weights + 16 int16 block sums. Used for intermediate quantization in dot
     * products. 9.125 bpw.
     */
    Q8_K(292, 256),

    // ============================================================
    // I-Quant (Inference-Optimized Quantization) Types
    // ============================================================

    /**
     * Layout: [{d: fp16, qs: uint16[32]}]. 1 fp16 scale + 32 uint16 storing 2-bit lookup table
     * indices (8 per uint16, 256 total). Uses 2×256 entry importance lookup tables. 2.0625 bpw.
     */
    IQ2_XXS(66, 256),

    /**
     * Layout: [{d: fp16, qs: uint16[32], scales: uint8[8]}]. 1 fp16 scale + 32 uint16 (2-bit table
     * indices) + 8 per-group scales. Uses 2×256 entry importance lookup tables. 2.3125 bpw.
     */
    IQ2_XS(74, 256),

    /**
     * Layout: [{d: fp16, qs: uint8[96]}]. 1 fp16 scale + 256 3-bit weights packed as 96 bytes (3
     * bytes per 8 weights). Uses 256-entry importance lookup table. 3.0625 bpw.
     */
    IQ3_XXS(98, 256),

    /**
     * Layout: [{d: fp16, qs: uint8[32], qh: uint16[8]}]. 1 fp16 scale + 256 1-bit values (qs, sign
     * + value packed as 2 bits per element in qh). Uses 16-entry importance lookup table. 1.5625
     * bpw.
     */
    IQ1_S(50, 256),

    /**
     * Layout: [{d: fp16, qs: uint8[16]}]. Same block shape as Q4_0 (1 fp16 scale + 32 packed 4-bit
     * weights) but uses non-linear importance-weighted quantization. 4.5 bpw.
     */
    IQ4_NL(18, 32),

    /**
     * Layout: [{d: fp16, qs: uint8[64], qh: uint8[8], signs: uint8[32], scales: uint8[4]}]. 16
     * sub-blocks of 16. qs = low 2 bits, qh = high bit, signs = sign bit = 3-bit signed weights. 4
     * bytes for 16 2-bit sub-block scales. 3.4375 bpw.
     */
    IQ3_S(110, 256),

    /**
     * Layout: [{d: fp16, qs: uint8[64], qh: uint8[8], scales: uint8[8]}]. 1 fp16 scale + 64 bytes
     * for 2-bit value + 8 bytes for sign bit + 8 bytes for 2-bit per-group scales. 2.5625 bpw.
     */
    IQ2_S(82, 256),

    /**
     * Layout: [{d: fp16, scales_h: uint16, scales_l: uint8[4], qs: uint8[128]}]. 1 fp16 scale + 6
     * bytes of 6-bit per-group scales + 128 bytes of 4-bit weights (256 elements). 4.25 bpw.
     */
    IQ4_XS(136, 256),

    // ============================================================
    // Integer Types
    // ============================================================

    /** 8-bit signed integer. 8.0 bpw. */
    I8(1, 1),

    /** 16-bit signed integer. 16.0 bpw. */
    I16(2, 1),

    /** 32-bit signed integer. 32.0 bpw. */
    I32(4, 1),

    /** 64-bit signed integer. 64.0 bpw. */
    I64(8, 1),

    /** 64-bit IEEE 754 double-precision floating point. 64.0 bpw. */
    F64(8, 1),

    /**
     * Layout: [{qs: uint8[32], qh: uint8[16], scales: uint8[8]}]. Grid-based lookup: qs = low 8
     * bits of grid index, qh = high 3 bits + grid shift bit, scales = 3-bit block scales. No fp16
     * scale (uses external iq1m_scale_t lookup table). 1.75 bpw.
     */
    IQ1_M(56, 256),

    /**
     * 16-bit Google Brain bfloat16 (Brain Floating Point). Has the same exponent range as FP32 but
     * reduced mantissa precision. 16.0 bpw.
     */
    BF16(2, 1),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    Q4_0_4_4(0, 0),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    Q4_0_4_8(0, 0),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    Q4_0_8_8(0, 0),

    /**
     * Layout: [{qs: uint8[50], qh: uint8[4], d: fp16}]. Ternary {-1, 0, +1} quantization: 252
     * elements packed as 5 per byte (3^5 = 243 < 256) in qs, 4 elements packed 1 bit each in qh, 1
     * fp16 scale. 1.6875 bpw.
     */
    TQ1_0(54, 256),

    /**
     * Layout: [{qs: uint8[64], d: fp16}]. 2-bit ternary quantization: 256 elements packed as 2 bits
     * each in qs[64], 1 fp16 scale. 2.0625 bpw.
     */
    TQ2_0(66, 256),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    IQ4_NL_4_4(0, 0),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    IQ4_NL_4_8(0, 0),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    IQ4_NL_8_8(0, 0),

    /**
     * Layout: [{e: uint8, qs: uint8[16]}]. 1 shared E8M0 exponent + 32 packed 4-bit E2M1 weights (2
     * per byte). Weight = 2^e × 2^m × v where m is mantissa, v is 0|0.5|1|1.5. 4.25 bpw.
     */
    MXFP4(17, 32),

    /**
     * Layout: [{d: uint8[4], qs: uint8[32]}]. 4 sub-blocks of 16 elements: 4 UE4M3 scales (1 per
     * sub-block) + 64 packed 4-bit E2M1 values (2 per byte). 4.5 bpw.
     */
    NVFP4(36, 64);

    // ============================================================
    // Fields
    // ============================================================

    /** Size in bytes for all elements in a block. */
    private final int blockByteSize;

    /** Number of elements per block. */
    private final int elementsPerBlock;

    // ============================================================
    // Constructor
    // ============================================================

    GGMLType(int blockByteSize, int elementsPerBlock) {
        if (blockByteSize == 0 && elementsPerBlock == 0) {
            this.blockByteSize = 0;
            this.elementsPerBlock = 0;
            return;
        }
        if (blockByteSize <= 0) {
            throw new IllegalArgumentException("blockByteSize must be positive: " + blockByteSize);
        }
        if (elementsPerBlock <= 0) {
            throw new IllegalArgumentException(
                    "elementsPerBlock must be positive: " + elementsPerBlock);
        }
        if (!isPowerOf2(elementsPerBlock)) {
            throw new IllegalArgumentException(
                    "elementsPerBlock must be a power of 2: " + elementsPerBlock);
        }
        this.blockByteSize = blockByteSize;
        this.elementsPerBlock = elementsPerBlock;
    }

    /** Cache of enum values to avoid creating new arrays on each call to {@link #values()}. */
    private static final GGMLType[] VALUES = values();

    // ============================================================
    // Public API
    // ============================================================

    /** Returns the GGML type ID (equivalent to ordinal in the {@code ggml_type} enum). */
    public int getId() {
        return ordinal();
    }

    /** Returns the size in bytes for all elements in a block. */
    public int getBlockByteSize() {
        return blockByteSize;
    }

    /**
     * Returns the number of elements per block.
     *
     * <p>For primitive types (F32, F16, I8, etc.), this returns 1. For quantized types, this is
     * typically 32 or 256.
     */
    public int getElementsPerBlock() {
        return elementsPerBlock;
    }

    /**
     * Returns whether this type is a quantized format. Quantized types have more than one element
     * per block.
     */
    public boolean isQuantized() {
        return elementsPerBlock > 1;
    }

    /**
     * Returns the effective bits per weight (BPW) for this type.
     *
     * <p>BPW is calculated as: {@code (blockByteSize * 8) / elementsPerBlock}
     *
     * <p>Examples:
     *
     * <ul>
     *   <li>F32: 32 bpw
     *   <li>Q4_0: 4.5 bpw
     *   <li>IQ2_XXS: 2.0625 bpw
     * </ul>
     */
    public double getBitsPerWeight() {
        return (blockByteSize * 8.0) / elementsPerBlock;
    }

    /**
     * Returns the GGMLType for the given type ID.
     *
     * @throws ArrayIndexOutOfBoundsException if the ID is invalid
     */
    public static GGMLType fromId(int id) {
        if (id < 0 || id >= VALUES.length) {
            throw new ArrayIndexOutOfBoundsException("Unknown GGML type ID: " + id);
        }
        return VALUES[id];
    }

    /**
     * Calculates the byte size required to store the given number of elements.
     *
     * @throws ArithmeticException if the result overflows
     * @throws IllegalArgumentException if element count is not a multiple of elements per block
     */
    public long byteSizeFor(long numberOfElements) {
        if (numberOfElements % elementsPerBlock != 0) {
            throw new IllegalArgumentException(
                    "Number of elements ("
                            + numberOfElements
                            + ") must be a multiple of elements per block ("
                            + elementsPerBlock
                            + ")");
        }
        long blocks = numberOfElements / elementsPerBlock;
        return Math.multiplyExact(blocks, blockByteSize);
    }

    /**
     * Calculates the number of elements that can be stored in the given byte size.
     *
     * @throws IllegalArgumentException if byte size is not a multiple of block byte size
     */
    public long elementsForByteSize(long byteSize) {
        if (byteSize % blockByteSize != 0) {
            throw new IllegalArgumentException(
                    "Byte size ("
                            + byteSize
                            + ") must be a multiple of block byte size ("
                            + blockByteSize
                            + ")");
        }
        long blocks = byteSize / blockByteSize;
        return blocks * elementsPerBlock;
    }

    // ============================================================
    // Private Helpers
    // ============================================================

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

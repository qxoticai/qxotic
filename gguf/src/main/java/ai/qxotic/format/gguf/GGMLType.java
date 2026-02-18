package ai.qxotic.format.gguf;

/**
 * GGML tensor data types.
 *
 * <p>Each type defines how tensor data is encoded in memory. Types fall into three categories:
 *
 * <ul>
 *   <li><b>Primitive types</b> - F32, F16, BF16, I8, I16, I32, I64, F64
 *   <li><b>Legacy quantization</b> - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
 *   <li><b>K-quantization</b> - Q2_K through Q8_K, IQ series, TQ series, MXFP4
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

    /**
     * 32-bit IEEE 754 single-precision floating point. Elements per block: 1, Block byte size: 4
     * bytes.
     */
    F32(4, 1),

    /**
     * 16-bit IEEE 754-2008 half-precision floating point (FP16). Elements per block: 1, Block byte
     * size: 2 bytes.
     */
    F16(2, 1),

    /**
     * 4-bit quantization with block-wise scale. Each 32-element block stores: 1 scale (FP16) + 32 ×
     * 4-bit weights packed into 16 bytes. Elements per block: 32, Block byte size: 18 bytes (4.5
     * bpw).
     */
    Q4_0(18, 32),

    /**
     * 4-bit quantization with block-wise scale and minimum. Each 32-element block stores: 1 scale
     * (FP16) + 1 min (FP16) + 32 × 4-bit weights. Elements per block: 32, Block byte size: 20 bytes
     * (5.0 bpw).
     */
    Q4_1(20, 32),

    /**
     * @deprecated Support has been removed from GGML.
     */
    @Deprecated
    Q4_2(Integer.MAX_VALUE, 1),

    /**
     * @deprecated Support has been removed from GGML.
     */
    @Deprecated
    Q4_3(Integer.MAX_VALUE, 1),

    /**
     * 5-bit quantization with block-wise scale. Each 32-element block stores: 1 scale (FP16) + 32 ×
     * 5-bit weights. The 5th bits are stored separately as a 32-bit bitmask (4 bytes). Elements per
     * block: 32, Block byte size: 22 bytes (5.5 bpw).
     */
    Q5_0(22, 32),

    /**
     * 5-bit quantization with block-wise scale and minimum. Each 32-element block stores: 1 scale
     * (FP16) + 1 min (FP16) + 32 × 5-bit weights. Elements per block: 32, Block byte size: 24 bytes
     * (6.0 bpw).
     */
    Q5_1(24, 32),

    /**
     * 8-bit quantization with block-wise scale. Each 32-element block stores: 1 scale (FP16) + 32 ×
     * 8-bit weights. Elements per block: 32, Block byte size: 34 bytes (8.5 bpw).
     */
    Q8_0(34, 32),

    /**
     * 8-bit quantization with block-wise scale and sum (for dot product optimization). Each
     * 32-element block stores: 1 scale (FP16) + 1 sum (FP16) + 32 × 8-bit weights. Elements per
     * block: 32, Block byte size: 36 bytes (9.0 bpw).
     */
    Q8_1(36, 32),

    // ============================================================
    // K-Quantization Types (elements per block = 256)
    // ============================================================

    /**
     * 2-bit K-quantization with super-block structure. Each 256-element super-block stores scales
     * and 2-bit quantized values. Elements per block: 256, Block byte size: 84 bytes (2.625 bpw).
     */
    Q2_K(84, 256),

    /**
     * 3-bit K-quantization with super-block structure. Elements per block: 256, Block byte size:
     * 110 bytes (3.4375 bpw).
     */
    Q3_K(110, 256),

    /**
     * 4-bit K-quantization with super-block structure. Elements per block: 256, Block byte size:
     * 144 bytes (4.5 bpw).
     */
    Q4_K(144, 256),

    /**
     * 5-bit K-quantization with super-block structure. Elements per block: 256, Block byte size:
     * 176 bytes (5.5 bpw).
     */
    Q5_K(176, 256),

    /**
     * 6-bit K-quantization with super-block structure. Elements per block: 256, Block byte size:
     * 210 bytes (6.5625 bpw).
     */
    Q6_K(210, 256),

    /**
     * 8-bit K-quantization with super-block structure (used for quantizing intermediate results).
     * Elements per block: 256, Block byte size: 292 bytes (9.125 bpw).
     */
    Q8_K(292, 256),

    // ============================================================
    // I-Quant (Inference-Optimized Quantization) Types
    // ============================================================

    /**
     * Importance-weighted quantization 2-bit, extra-extra-small. Uses lookup tables for very compact 2-bit
     * representation. Elements per block: 256, Block byte size: 66 bytes (2.0625 bpw).
     */
    IQ2_XXS(66, 256),

    /**
     * Importance-weighted quantization 2-bit, extra-small. Elements per block: 256, Block byte size: 74 bytes
     * (2.3125 bpw).
     */
    IQ2_XS(74, 256),

    /**
     * Importance-weighted quantization 3-bit, extra-extra-small. Elements per block: 256, Block byte size: 98
     * bytes (3.0625 bpw).
     */
    IQ3_XXS(98, 256),

    /**
     * Importance-weighted quantization 1-bit, small. Elements per block: 256, Block byte size: 50 bytes
     * (1.5625 bpw).
     */
    IQ1_S(50, 256),

    /**
     * Importance-weighted quantization 4-bit, non-linear. Uses a 32-element block like Q4_0 but with
     * non-linear quantization. Elements per block: 32, Block byte size: 18 bytes (4.5 bpw).
     */
    IQ4_NL(18, 32),

    /**
     * Importance-weighted quantization 3-bit, small. Elements per block: 256, Block byte size: 110 bytes
     * (3.4375 bpw).
     */
    IQ3_S(110, 256),

    /**
     * Importance-weighted quantization 2-bit, small. Elements per block: 256, Block byte size: 82 bytes
     * (2.5625 bpw).
     */
    IQ2_S(82, 256),

    /**
     * Importance-weighted quantization 4-bit, extra-small. Elements per block: 256, Block byte size: 136
     * bytes (4.25 bpw).
     */
    IQ4_XS(136, 256),

    // ============================================================
    // Integer Types
    // ============================================================

    /** 8-bit signed integer. Elements per block: 1, Block byte size: 1 byte. */
    I8(1, 1),

    /** 16-bit signed integer. Elements per block: 1, Block byte size: 2 bytes. */
    I16(2, 1),

    /** 32-bit signed integer. Elements per block: 1, Block byte size: 4 bytes. */
    I32(4, 1),

    /** 64-bit signed integer. Elements per block: 1, Block byte size: 8 bytes. */
    I64(8, 1),

    /**
     * 64-bit IEEE 754 double-precision floating point. Elements per block: 1, Block byte size: 8
     * bytes.
     */
    F64(8, 1),

    /**
     * Importance-weighted quantization 1-bit, medium. Elements per block: 256, Block byte size: 56 bytes
     * (1.75 bpw).
     */
    IQ1_M(56, 256),

    /**
     * 16-bit Google Brain bfloat16 (Brain Floating Point). Has the same exponent range as FP32 but
     * reduced mantissa precision. Elements per block: 1, Block byte size: 2 bytes.
     */
    BF16(2, 1),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    Q4_0_4_4(18, 32),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    Q4_0_4_8(18, 32),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    Q4_0_8_8(18, 32),

    /**
     * Ternary Quantization 1-bit. Stores weights as {-1, 0, +1} values. Elements per block: 256,
     * Block byte size: 54 bytes (1.6875 bpw).
     */
    TQ1_0(54, 256),

    /**
     * Ternary Quantization 2-bit. Elements per block: 256, Block byte size: 66 bytes (2.0625 bpw).
     */
    TQ2_0(66, 256),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    IQ4_NL_4_4(18, 32),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    IQ4_NL_4_8(18, 32),

    /**
     * @deprecated Support has been removed from GGUF files.
     */
    @Deprecated
    IQ4_NL_8_8(18, 32),

    /**
     * Microscaling Format 4-bit (MXFP4) with E8M0 shared exponent. Uses 1-byte E8M0 exponent + 16
     * bytes of 4-bit weights. Elements per block: 32, Block byte size: 17 bytes (4.25 bpw).
     */
    MXFP4(17, 32);

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

    // ============================================================
    // Public API
    // ============================================================

    /**
     * Returns the GGML type ID (equivalent to ordinal in the {@code ggml_type} enum).
     *
     * @return the type ID (0-based index)
     */
    public int getId() {
        return ordinal();
    }

    /**
     * Returns the size in bytes for all elements in a block.
     *
     * @return the block byte size
     */
    public int getBlockByteSize() {
        return blockByteSize;
    }

    /**
     * Returns the number of elements per block.
     *
     * <p>For primitive types (F32, F16, I8, etc.), this returns 1. For quantized types, this is
     * typically 32 or 256.
     *
     * @return the number of elements per block
     */
    public int getElementsPerBlock() {
        return elementsPerBlock;
    }

    /**
     * Returns whether this type is a quantized format. Quantized types have more than one element
     * per block.
     *
     * @return {@code true} if this is a quantized type
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
     *
     * @return the bits per weight
     */
    public double getBitsPerWeight() {
        return (blockByteSize * 8.0) / elementsPerBlock;
    }

    /**
     * Returns the GGMLType for the given type ID.
     *
     * @param id the GGML type ID (0-based index)
     * @return the corresponding GGMLType
     * @throws ArrayIndexOutOfBoundsException if the ID is invalid
     */
    public static GGMLType fromId(int id) {
        GGMLType[] values = values();
        if (id < 0 || id >= values.length) {
            throw new ArrayIndexOutOfBoundsException("Unknown GGML type ID: " + id);
        }
        return values[id];
    }

    /**
     * Calculates the byte size required to store a tensor of this type with the given shape.
     *
     * @param shape the tensor dimensions
     * @return the byte size required
     * @throws ArithmeticException if the result overflows
     * @throws IllegalArgumentException if element count is not a multiple of elements per block
     */
    public long byteSizeForShape(long[] shape) {
        long elementCount = 1;
        for (long dim : shape) {
            elementCount = Math.multiplyExact(elementCount, dim);
        }
        return byteSizeFor(elementCount);
    }

    /**
     * Calculates the byte size required to store the given number of elements.
     *
     * @param numberOfElements the total number of elements
     * @return the byte size required
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
     * @param byteSize the available byte size
     * @return the number of elements that can be stored
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

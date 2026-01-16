package ai.qxotic.format.gguf;

import java.util.Arrays;

public enum GGMLType {
    F32(Float.BYTES),
    F16(GGMLType.FLOAT16_BYTES),
    Q4_0(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_1),

    Q4_2(Integer.MAX_VALUE, 1, false, true), // support has been removed
    Q4_3(Integer.MAX_VALUE, 1, false, true), // support has been removed

    Q5_0(GGMLType.FLOAT16_BYTES + 4 + 16, GGMLType.QK5_0),
    Q5_1(2 * GGMLType.FLOAT16_BYTES + 4 + 16, GGMLType.QK5_1),
    Q8_0(GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, GGMLType.QK8_0),
    Q8_1(2 * GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, GGMLType.QK8_1),
    // k-quantizations
    Q2_K(16 + 64 + 4, GGMLType.QK_K),
    Q3_K(32 + 64 + 12 + 2, GGMLType.QK_K),
    Q4_K(
            2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2,
            GGMLType.QK_K),
    Q5_K(
            2 * GGMLType.FLOAT16_BYTES
                    + ((GGMLType.QK_K / 16) / 8 * 6)
                    + GGMLType.QK_K / 8
                    + GGMLType.QK_K / 2,
            GGMLType.QK_K),
    Q6_K(
            GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES,
            GGMLType.QK_K),
    Q8_K(4 + 256 + 32, GGMLType.QK_K),

    IQ2_XXS(2 + 64, GGMLType.QK_K),
    IQ2_XS(2 + 64 + 8, GGMLType.QK_K),
    IQ3_XXS(2 + 96, GGMLType.QK_K),
    IQ1_S(2 + 32 + 16, GGMLType.QK_K),
    IQ4_NL(2 + 16, GGMLType.QK4_NL),
    IQ3_S(2 + 64 + 8 + 32 + 4, GGMLType.QK_K),
    IQ2_S(2 + 64 + 8 + 8, GGMLType.QK_K),
    IQ4_XS(2 + 2 + 4 + 128, GGMLType.QK_K),

    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(32 + 16 + 8, GGMLType.QK_K),
    BF16(GGMLType.BFLOAT16_BYTES),
    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    TQ1_0(51 + 4 + 2, GGMLType.QK_K),
    TQ2_0(64 + 2, GGMLType.QK_K),
    IQ4_NL_4_4(2 + 16, GGMLType.QK4_NL),
    IQ4_NL_4_8(2 + 16, GGMLType.QK4_NL),
    IQ4_NL_8_8(2 + 16, GGMLType.QK4_NL),
    MXFP4(Byte.BYTES + 16, GGMLType.QK_MXFP4);

    private static final int FLOAT16_BYTES = 2;
    private static final int BFLOAT16_BYTES = 2;

    private static final GGMLType[] VALUES = values();

    private final int blockByteSize;
    private final int elementsPerBlock;
    private final boolean isQuantized;
    private final boolean isDeprecated;

    public int getBlockByteSize() {
        return blockByteSize;
    }

    public int getElementsPerBlock() {
        return elementsPerBlock;
    }

    public boolean isQuantized() {
        return isQuantized;
    }

    public boolean isDeprecated() {
        return isDeprecated;
    }

    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    public long byteSizeForShape(long[] shape) {
        long elementCount = Arrays.stream(shape).reduce(1L, Math::multiplyExact);
        return byteSizeFor(elementCount);
    }

    public long byteSizeFor(long numberOfElements) {
        long t = Math.multiplyExact(numberOfElements, getBlockByteSize());
        assert t % getElementsPerBlock() == 0;
        return t / getElementsPerBlock();
    }

    private static final int QK_K = 256;
    private static final int QK4_0 = 32;
    private static final int QK4_1 = 32;
    private static final int QK5_0 = 32;
    private static final int QK5_1 = 32;
    private static final int QK8_0 = 32;
    private static final int QK8_1 = 32;
    private static final int QK4_NL = 32;
    private static final int QK_MXFP4 = 32;

    GGMLType(int blockByteSize) {
        this(blockByteSize, 1, false, false);
    }

    GGMLType(int blockByteSize, int elementsPerBlock) {
        this(blockByteSize, elementsPerBlock, true, false);
    }

    GGMLType(int blockByteSize, int elementsPerBlock, boolean isQuantized) {
        this(blockByteSize, elementsPerBlock, isQuantized, false);
    }

    GGMLType(int blockByteSize, int elementsPerBlock, boolean isQuantized, boolean isDeprecated) {
        assert elementsPerBlock > 0;
        assert blockByteSize > 0;
        assert isPowerOf2(elementsPerBlock);
        this.blockByteSize = blockByteSize;
        this.elementsPerBlock = elementsPerBlock;
        this.isQuantized = isQuantized;
        this.isDeprecated = isDeprecated;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

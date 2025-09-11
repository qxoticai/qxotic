package com.llm4j.gguf;

import com.llm4j.api.BaseType;

public enum GGMLType implements BaseType {
    F32(Float.BYTES),
    F16(GGMLType.FLOAT16_BYTES),
    Q4_0(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_1),

    Q4_2(Integer.MAX_VALUE), // support has been removed
    Q4_3(Integer.MAX_VALUE), // support has been removed

    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, GGMLType.QK8_0),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, GGMLType.QK8_0),
    // k-quantizations
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(
            2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2,
            GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),

    IQ2_XXS(Integer.MAX_VALUE),
    IQ2_XS(Integer.MAX_VALUE),
    IQ3_XXS(Integer.MAX_VALUE),
    IQ1_S(Integer.MAX_VALUE),
    IQ4_NL(Integer.MAX_VALUE),
    IQ3_S(Integer.MAX_VALUE),
    IQ2_S(Integer.MAX_VALUE),
    IQ4_XS(Integer.MAX_VALUE),

    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(Integer.MAX_VALUE),
    BF16(GGMLType.BFLOAT16_BYTES),
    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, GGMLType.QK4_0),
    TQ1_0(Integer.MAX_VALUE),
    TQ2_0(Integer.MAX_VALUE);

    private static final int FLOAT16_BYTES = 2;
    private static final int BFLOAT16_BYTES = 2;

    private static final GGMLType[] VALUES = values();

    private final int blockByteSize;
    private final int elementsPerBlock;
    private final boolean isQuantized;

    public int getBlockByteSize() {
        return blockByteSize;
    }

    public int getElementsPerBlock() {
        return elementsPerBlock;
    }

    @Override
    public boolean isQuantized() {
        return isQuantized;
    }

    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    @Override
    public long byteSizeFor(long numberOfElements) {
        long t = numberOfElements * getBlockByteSize();
        assert t % getElementsPerBlock() == 0;
        return t / getElementsPerBlock();
    }

    public static final int QK_K = 256; // or 64?

    private static final int QK4_0 = 32;
    private static final int QK4_1 = 32;
    private static final int QK8_0 = 32;

    GGMLType(int blockByteSize) {
        this(blockByteSize, 1, false);
    }

    GGMLType(int blockByteSize, int elementsPerBlock) {
        this(blockByteSize, elementsPerBlock, true);
    }

    GGMLType(int blockByteSize, int elementsPerBlock, boolean isQuantized) {
        assert elementsPerBlock > 0;
        assert blockByteSize > 0;
        assert isPowerOf2(elementsPerBlock);
        this.blockByteSize = blockByteSize;
        this.elementsPerBlock = elementsPerBlock;
        this.isQuantized = isQuantized;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

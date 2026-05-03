package com.qxotic.format.safetensors;

import java.util.Arrays;
import java.util.Objects;

/**
 * Data types supported by the Safetensors format. Must be declared in increasing alignment order to
 * match the upstream specification.
 *
 * @see <a
 *     href="https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs">Upstream
 *     DType</a>
 */
public enum DType {
    // Boolean type
    BOOL(8, boolean.class),
    // MXF4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
    F4(4, byte.class),
    // MXF6 E2M3
    F6_E2M3(6, byte.class),
    // MXF6 E3M2
    F6_E3M2(6, byte.class),
    // Unsigned byte
    U8(8, byte.class),
    // Signed byte
    I8(8, byte.class),
    // FP8 E5M2 <https://arxiv.org/pdf/2209.05433.pdf>
    F8_E5M2(8, byte.class),
    // FP8 E4M3
    F8_E4M3(8, byte.class),
    // FP8 E8M0
    // <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
    F8_E8M0(8, byte.class),
    // FP8 E4M3 FNUZ variant
    F8_E4M3FNUZ(8, byte.class),
    // FP8 E5M2 FNUZ variant
    F8_E5M2FNUZ(8, byte.class),
    // Signed integer (16-bit)
    I16(16, short.class),
    // Unsigned integer (16-bit)
    U16(16, short.class),
    // Half-precision floating point
    F16(16, short.class),
    // Brain floating point
    BF16(16, short.class),
    // Signed integer (32-bit)
    I32(32, int.class),
    // Unsigned integer (32-bit)
    U32(32, int.class),
    // Floating point (32-bit)
    F32(32, float.class),
    // Complex (64-bit, 2×F32)
    C64(64, float.class),
    // Floating point (64-bit)
    F64(64, double.class),
    // Signed integer (64-bit)
    I64(64, long.class),
    // Unsigned integer (64-bit)
    U64(64, long.class);

    private final int bitSize;
    private final Class<?> javaType;

    DType(int bitSize, Class<?> javaType) {
        this.bitSize = bitSize;
        this.javaType = javaType;
    }

    /** Number of bits per element. */
    public int bitSize() {
        return bitSize;
    }

    /**
     * Java primitive carrier storage type for this dtype. For sub-byte and FP8 types, returns
     * byte.class.
     */
    public Class<?> javaType() {
        return javaType;
    }

    /** Payload byte size for a flat element count using bitsize semantics. */
    public long byteSizeFor(long numElements) {
        long totalBits = Math.multiplyExact(numElements, (long) bitSize);
        if (totalBits % 8 != 0) {
            throw new IllegalArgumentException(
                    "Misaligned slice: "
                            + name()
                            + " tensor with "
                            + numElements
                            + " elements occupies "
                            + totalBits
                            + " bits ("
                            + (totalBits / 8.0)
                            + " bytes)");
        }
        return totalBits / 8;
    }

    /** Payload byte size for a tensor shape. Throws on overflow or misaligned sub-byte shapes. */
    public long byteSizeForShape(long[] shape) {
        Objects.requireNonNull(shape, "shape");
        long numElements = Arrays.stream(shape).reduce(1L, Math::multiplyExact);
        return byteSizeFor(numElements);
    }
}

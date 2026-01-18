package ai.qxotic.format.safetensors;

import java.util.Arrays;

/** Data types supported by the Safetensors format. */
public enum DType {
    // Boolean type
    BOOL(1, boolean.class),
    // Unsigned byte
    U8(1, byte.class),
    // Signed byte
    I8(1, byte.class),
    // Signed integer (16-bit)
    I16(2, short.class),
    // Unsigned integer (16-bit)
    U16(2, short.class),
    // Half-precision floating point
    F16(2, short.class), // Java has not float16
    // Brain floating point
    BF16(2, short.class), // Java has not bfloat16
    // Signed integer (32-bit)
    I32(4, int.class),
    // Unsigned integer (32-bit)
    U32(4, int.class),
    // Floating point (32-bit)
    F32(4, float.class),
    // Floating point (64-bit)
    F64(8, double.class),
    // Signed integer (64-bit)
    I64(8, long.class),
    // Unsigned integer (64-bit)
    U64(8, long.class);

    private final int size;

    public Class<?> javaType() {
        return javaType;
    }

    private final Class<?> javaType;

    DType(int size, Class<?> javaType) {
        this.size = size;
        this.javaType = javaType;
    }

    public int size() {
        return size;
    }

    public long byteSizeFor(long numElements) {
        return Math.multiplyExact(numElements, (long) size);
    }

    public long byteSizeForShape(long[] shape) {
        return byteSizeFor(totalNumberOfElements(shape));
    }

    public static long totalNumberOfElements(long[] shape) {
        return Arrays.stream(shape).reduce(1L, Math::multiplyExact);
    }
}

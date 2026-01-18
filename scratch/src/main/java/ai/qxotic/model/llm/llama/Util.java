package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGMLType;
import ai.qxotic.span.DirectAccessOps;
import ai.qxotic.span.FloatSpan;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.util.stream.LongStream;
import sun.misc.Unsafe;

public class Util {
    public static long numberOfElements(long... dims) {
        assert LongStream.of(dims).allMatch(d -> d > 0);
        return LongStream.of(dims).reduce(1, Math::multiplyExact);
    }

    private static final ClassValue<GGMLType> ggmlType =
            new ClassValue<>() {
                @Override
                protected GGMLType computeValue(Class<?> type) {
                    if (type == F32Span.class || type == ArraySpan.class) {
                        return GGMLType.F32;
                    }
                    if (type == Q8_0Span.class || type == Q8_0BBSpan.class) {
                        return GGMLType.Q8_0;
                    }
                    if (type == Q4_0Span.class || type == Q4_0BBSpan.class) {
                        return GGMLType.Q4_0;
                    }
                    if (type == Q4_1Span.class) {
                        return GGMLType.Q4_1;
                    }
                    if (type == BF16Span.class) {
                        return GGMLType.BF16;
                    }
                    throw new IllegalArgumentException("Unknown GGML type for " + type);
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> F32SpanDirectAccess =
            new DirectAccessOps<F32Span>() {
                @Override
                public float getElementAt(F32Span span, long index) {
                    // return this.memorySegment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
                    return readFloat(span.memorySegment, index * Float.BYTES);
                }

                @Override
                public void setElementAt(F32Span span, long index, float value) {
                    // this.memorySegment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
                    writeFloat(span.memorySegment, index * Float.BYTES, value);
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> BF16DirectAccess =
            new DirectAccessOps<BF16Span>() {
                @Override
                public float getElementAt(BF16Span span, long index) {
                    // return
                    // BFloat16.bfloat16ToFloat(span.memorySegment.getAtIndex(ValueLayout.JAVA_SHORT, index));
                    return BFloat16.bfloat16ToFloat(
                            readShort(span.memorySegment, index * BFloat16.BYTES));
                }

                @Override
                public void setElementAt(BF16Span span, long index, float value) {
                    // span.memorySegment.setAtIndex(ValueLayout.JAVA_SHORT, index,
                    // BFloat16.floatToBFloat16(value));
                    writeShort(
                            span.memorySegment,
                            index * BFloat16.BYTES,
                            BFloat16.floatToBFloat16(value));
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> F32BBSpanDirectAccess =
            new DirectAccessOps<F32BBSpan>() {
                @Override
                public float getElementAt(F32BBSpan span, long index) {
                    return span.byteBuffer.getFloat((int) index * Float.BYTES);
                }

                @Override
                public void setElementAt(F32BBSpan span, long index, float value) {
                    span.byteBuffer.putFloat((int) index * Float.BYTES, value);
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> ArraySpanDirectAccess =
            new DirectAccessOps<ArraySpan>() {
                @Override
                public float getElementAt(ArraySpan span, long index) {
                    checkBounds(0 <= index && index < span.size(), "index out of bounds");
                    return span.values[(int) (span.offset + index)];
                }

                @Override
                public void setElementAt(ArraySpan span, long index, float value) {
                    checkBounds(0 <= index && index < span.size(), "index out of bounds");
                    span.values[(int) (span.offset + index)] = value;
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> Q8_0DirectAccess =
            new DirectAccessOps<Q8_0Span>() {
                @Override
                public float getElementAt(Q8_0Span span, long index) {
                    assert 0 <= index && index < span.size();
                    long blockIndex = index / GGMLType.Q8_0.getElementsPerBlock();
                    long withinBlockIndex = index % GGMLType.Q8_0.getElementsPerBlock();
                    long blockOffset = blockIndex * GGMLType.Q8_0.getBlockByteSize();
                    byte quant =
                            readByte(
                                    span.memorySegment,
                                    blockOffset + Float16.BYTES + withinBlockIndex);
                    float scale = Float.float16ToFloat(readShort(span.memorySegment, blockOffset));
                    return quant * scale;
                }

                @Override
                public void setElementAt(Q8_0Span span, long index, float value) {
                    throw new UnsupportedOperationException();
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> Q8_0BBDirectAccess =
            new DirectAccessOps<Q8_0BBSpan>() {
                @Override
                public float getElementAt(Q8_0BBSpan span, long longIndex) {
                    assert 0 <= longIndex && longIndex < span.size();
                    int index = (int) longIndex;
                    int blockIndex = index / GGMLType.Q8_0.getElementsPerBlock();
                    int withinBlockIndex = index % GGMLType.Q8_0.getElementsPerBlock();
                    int blockOffset = blockIndex * GGMLType.Q8_0.getBlockByteSize();
                    byte quant =
                            span.byteBuffer.get(blockOffset + Float16.BYTES + withinBlockIndex);
                    float scale = Float.float16ToFloat(span.byteBuffer.getShort(blockOffset));
                    return quant * scale;
                }

                @Override
                public void setElementAt(Q8_0BBSpan span, long index, float value) {
                    throw new UnsupportedOperationException();
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> Q4_0DirectAccess =
            new DirectAccessOps<Q4_0Span>() {
                @Override
                public float getElementAt(Q4_0Span span, long index) {
                    assert 0 <= index && index < span.size();
                    long blockIndex = index / GGMLType.Q4_0.getElementsPerBlock();
                    long blockOffset = blockIndex * GGMLType.Q4_0.getBlockByteSize();
                    float scale = Float.float16ToFloat(readShort(span.memorySegment, blockOffset));
                    byte quant;
                    long modIndex = index % GGMLType.Q4_0.getElementsPerBlock();
                    if (modIndex < GGMLType.Q4_0.getElementsPerBlock() / 2) {
                        quant =
                                (byte)
                                        (readByte(
                                                        span.memorySegment,
                                                        blockOffset + Float16.BYTES + modIndex)
                                                & 0x0F);
                    } else {
                        quant =
                                (byte)
                                        ((readByte(
                                                                span.memorySegment,
                                                                blockOffset
                                                                        + Float16.BYTES
                                                                        + modIndex
                                                                        - GGMLType.Q4_0
                                                                                        .getElementsPerBlock()
                                                                                / 2)
                                                        >>> 4)
                                                & 0x0F);
                    }
                    quant -= 8;
                    return quant * scale;
                }

                @Override
                public void setElementAt(Q4_0Span span, long index, float value) {
                    throw new UnsupportedOperationException();
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> Q4_1DirectAccess =
            new DirectAccessOps<Q4_1Span>() {
                @Override
                public float getElementAt(Q4_1Span span, long index) {
                    assert 0 <= index && index < span.size();
                    long blockIndex = index / GGMLType.Q4_1.getElementsPerBlock();
                    long blockOffset = blockIndex * GGMLType.Q4_1.getBlockByteSize();
                    float scale = Float.float16ToFloat(readShort(span.memorySegment, blockOffset));
                    float offset =
                            Float.float16ToFloat(
                                    readShort(span.memorySegment, blockOffset + Float16.BYTES));
                    byte quant;
                    long modIndex = index % GGMLType.Q4_1.getElementsPerBlock();
                    if (modIndex < GGMLType.Q4_1.getElementsPerBlock() / 2) {
                        quant =
                                (byte)
                                        (readByte(
                                                        span.memorySegment,
                                                        blockOffset
                                                                + Float16.BYTES
                                                                + Float16.BYTES
                                                                + modIndex)
                                                & 0x0F);
                    } else {
                        quant =
                                (byte)
                                        ((readByte(
                                                                span.memorySegment,
                                                                blockOffset
                                                                        + Float16.BYTES
                                                                        + Float16.BYTES
                                                                        + modIndex
                                                                        - GGMLType.Q4_1
                                                                                        .getElementsPerBlock()
                                                                                / 2)
                                                        >>> 4)
                                                & 0x0F);
                    }
                    return quant * scale + offset;
                }

                @Override
                public void setElementAt(Q4_1Span span, long index, float value) {
                    throw new UnsupportedOperationException();
                }
            };

    private static final DirectAccessOps<? extends FloatSpan> Q4_0BBDirectAccess =
            new DirectAccessOps<Q4_0BBSpan>() {
                @Override
                public float getElementAt(Q4_0BBSpan span, long longIndex) {
                    assert 0 <= longIndex && longIndex < span.size();
                    int index = (int) longIndex;
                    int blockIndex = index / GGMLType.Q4_0.getElementsPerBlock();
                    int blockOffset = blockIndex * GGMLType.Q4_0.getBlockByteSize();
                    float scale = Float.float16ToFloat(span.byteBuffer.getShort(blockOffset));
                    byte quant;
                    int modIndex = index % GGMLType.Q4_0.getElementsPerBlock();
                    if (modIndex < GGMLType.Q4_0.getElementsPerBlock() / 2) {
                        quant =
                                (byte)
                                        (span.byteBuffer.get(blockOffset + Float16.BYTES + modIndex)
                                                & 0x0F);
                    } else {
                        quant =
                                (byte)
                                        ((span.byteBuffer.get(
                                                                blockOffset
                                                                        + Float16.BYTES
                                                                        + modIndex
                                                                        - GGMLType.Q4_0
                                                                                        .getElementsPerBlock()
                                                                                / 2)
                                                        >>> 4)
                                                & 0x0F);
                    }
                    quant -= 8;
                    return quant * scale;
                }

                @Override
                public void setElementAt(Q4_0BBSpan span, long index, float value) {
                    throw new UnsupportedOperationException();
                }
            };

    private static final ClassValue<DirectAccessOps<? extends FloatSpan>> DIRECT_ACCESS =
            new ClassValue<>() {
                @Override
                protected DirectAccessOps<? extends FloatSpan> computeValue(Class<?> type) {
                    // float[]-backed span.
                    if (type == ArraySpan.class) {
                        return ArraySpanDirectAccess;
                    }

                    // MemorySegment-backed spans.
                    if (type == F32Span.class) {
                        return F32SpanDirectAccess;
                    }
                    if (type == Q8_0Span.class) {
                        return Q8_0DirectAccess;
                    }
                    if (type == Q4_0Span.class) {
                        return Q4_0DirectAccess;
                    }
                    if (type == Q4_1Span.class) {
                        return Q4_1DirectAccess;
                    }
                    if (type == BF16Span.class) {
                        return BF16DirectAccess;
                    }

                    // ByteBuffer-backed spans.
                    if (type == F32BBSpan.class) {
                        return F32BBSpanDirectAccess;
                    }
                    if (type == Q8_0BBSpan.class) {
                        return Q8_0BBDirectAccess;
                    }
                    if (type == Q4_0BBSpan.class) {
                        return Q4_0BBDirectAccess;
                    }
                    throw new IllegalArgumentException("No DirectAccessOps impl for  " + type);
                }
            };

    public static DirectAccessOps<FloatSpan> directAccess(FloatSpan span) {
        return (DirectAccessOps<FloatSpan>) DIRECT_ACCESS.get(span.getClass());
    }

    public static void checkBounds(boolean condition, String message) {
        if (!condition) {
            throw new IndexOutOfBoundsException(message);
        }
    }

    // The use of Unsafe in this file is a temporary workaround to support native-image.
    private static final Unsafe UNSAFE;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static short readShort(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        assert 0 <= offset && offset <= memorySegment.byteSize() - Short.BYTES;
        // return memorySegment.get(ValueLayout.JAVA_SHORT, offset);
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    public static byte readByte(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        assert 0 <= offset && offset <= memorySegment.byteSize() - Byte.BYTES;
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    public static float readFloat(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        assert 0 <= offset && offset <= memorySegment.byteSize() - Float.BYTES;
        return UNSAFE.getFloat(memorySegment.address() + offset);
    }

    public static void writeFloat(MemorySegment memorySegment, long offset, float value) {
        // The MemorySegment.get* methods should be used instead.
        assert 0 <= offset && offset <= memorySegment.byteSize() - Float.BYTES;
        UNSAFE.putFloat(memorySegment.address() + offset, value);
    }

    public static void writeShort(MemorySegment memorySegment, long offset, short value) {
        // The MemorySegment.get* methods should be used instead.
        assert 0 <= offset && offset <= memorySegment.byteSize() - Short.BYTES;
        UNSAFE.putShort(memorySegment.address() + offset, value);
    }
}

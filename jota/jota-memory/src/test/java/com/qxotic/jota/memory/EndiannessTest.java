package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Endianness contract tests.
 *
 * <p>Every {@code MemoryAccess} in jota-memory interprets multi-byte values in the platform's
 * native byte order: the {@code MemorySegment} domain uses the FFM {@code JAVA_*} layouts (which
 * are native order per their spec), the heap {@code byte[]}/{@code float[]} domains use {@code
 * sun.misc.Unsafe} (native), and the {@code ByteBuffer} domain uses {@code
 * ByteOrder.nativeOrder()}. Cross-domain copies are byte-preserving ({@code
 * MemorySegment.copy}/{@code System.arraycopy}), so a value written in one domain must read back
 * identically in every other domain, and the raw bytes must be native order.
 *
 * <p>These tests use deliberately asymmetric bit patterns (not {@code 0.0f}/{@code 1.0f}, whose
 * reversed bytes happen to round-trip under a byte-swap, masking order bugs).
 */
class EndiannessTest extends AbstractMemoryTest {

    /**
     * Asymmetric 32-bit float bit patterns: the low byte of each half differs from the high byte.
     */
    private static final int[] ASYMMETRIC_F32_BITS = {
        Float.floatToIntBits(1.5f), // 0x3FC00000
        Float.floatToIntBits(-2.0f), // 0xC0000000
        Float.floatToIntBits(123.456f), // 0x42F6E979
        Float.floatToIntBits(0.0001f), // 0x38D1B717
    };

    private static final int[] ASYMMETRIC_I32 = {0x01020304, 0xDEADBEEF, 0x7FFFFFFF, -123456789};

    private static final long[] ASYMMETRIC_I64 = {
        0x0102030405060708L, 0xDEADBEEFCAFEBABEL, Long.MIN_VALUE, -987654321012345678L
    };

    static Stream<MemoryDomain<?>> f32Domains() {
        return Stream.of(
                MemoryDomains.bytes(),
                MemoryDomains.floats(),
                MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false)),
                MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true)),
                MemoryDomains.of(MemoryAllocators.newScopedArena()));
    }

    // ---- 1. Cross-domain copies are byte-faithful for asymmetric values ----

    @ParameterizedTest
    @MethodSource("f32Domains")
    <B> void floatCopyBetweenEveryPairOfDomainsPreservesValues(MemoryDomain<B> domain) {
        List<MemoryDomain<?>> others = f32Domains().toList();
        for (MemoryDomain<?> other : others) {
            crossDomainFloatRoundTrip(domain, other);
        }
    }

    private static <B, C> void crossDomainFloatRoundTrip(
            MemoryDomain<B> left, MemoryDomain<C> right) {
        float[] values = new float[ASYMMETRIC_F32_BITS.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = Float.intBitsToFloat(ASYMMETRIC_F32_BITS[i]);
        }

        MemoryView<B> src = viewOf(left, values);
        MemoryView<C> mid = allocate(right, DataType.FP32, Shape.of(values.length));
        MemoryDomain.copy(left, src, right, mid);

        assertArrayEquals(values, baseFloats(right, mid), 0.0f);

        MemoryView<B> back = allocate(left, DataType.FP32, Shape.of(values.length));
        MemoryDomain.copy(right, mid, left, back);
        assertArrayEquals(
                values,
                baseFloats(left, back),
                0.0f,
                "float[] values changed after round-trip through "
                        + right.device()
                        + " ("
                        + right.memoryAllocator().getClass().getSimpleName()
                        + ")");
    }

    @Test
    void intAndLongCopyBetweenDomainsPreservesValues() {
        MemoryDomain<float[]> floats = MemoryDomains.floats();
        MemoryDomain<byte[]> bytes = MemoryDomains.bytes();
        MemoryDomain<MemorySegment> seg = MemoryDomains.of(MemoryAllocators.newScopedArena());

        MemoryView<float[]> intsInFloats =
                MemoryView.of(
                        floats.memoryAllocator().allocateMemory(DataType.I32, Shape.of(2)),
                        DataType.I32,
                        Layout.rowMajor(Shape.of(2)));
        MemoryView<MemorySegment> intsInSeg =
                MemoryView.of(
                        seg.memoryAllocator().allocateMemory(DataType.I32, Shape.of(2)),
                        DataType.I32,
                        Layout.rowMajor(Shape.of(2)));
        MemoryAccess<MemorySegment> segAccess = seg.directAccess();
        for (int i = 0; i < 2; i++) {
            segAccess.writeInt(intsInSeg.memory(), (long) i * Integer.BYTES, ASYMMETRIC_I32[i]);
        }
        MemoryDomain.copy(seg, intsInSeg, floats, intsInFloats);
        MemoryAccess<float[]> floatAccess = floats.directAccess();
        for (int i = 0; i < 2; i++) {
            assertEquals(
                    ASYMMETRIC_I32[i],
                    floatAccess.readInt(intsInFloats.memory(), (long) i * Integer.BYTES),
                    "int " + i + " through float[] domain");
        }

        // long[] value -> byte[] domain, read back via its long accessor
        MemoryView<byte[]> longsInBytes =
                MemoryView.of(
                        bytes.memoryAllocator().allocateMemory(DataType.I64, Shape.of(1)),
                        DataType.I64,
                        Layout.rowMajor(Shape.of(1)));
        MemoryAccess<byte[]> byteAccess = bytes.directAccess();
        MemoryAccess<MemorySegment> segAccess2 = seg.directAccess();
        MemoryView<MemorySegment> longsInSeg =
                MemoryView.of(
                        seg.memoryAllocator().allocateMemory(DataType.I64, Shape.of(1)),
                        DataType.I64,
                        Layout.rowMajor(Shape.of(1)));
        segAccess2.writeLong(longsInSeg.memory(), 0, ASYMMETRIC_I64[0]);
        MemoryDomain.copy(seg, longsInSeg, bytes, longsInBytes);
        assertEquals(
                ASYMMETRIC_I64[0],
                byteAccess.readLong(longsInBytes.memory(), 0),
                "long through byte[] domain");
    }

    @Test
    void copiesBetweenHeapAndDirectByteBuffers() {
        MemoryDomain<ByteBuffer> heap =
                MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false));
        MemoryDomain<ByteBuffer> direct =
                MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(true));

        float[] values = {1.5f, -2.0f, 123.456f};
        MemoryView<ByteBuffer> src = viewOf(heap, values);
        MemoryView<ByteBuffer> dst =
                MemoryView.of(
                        direct.memoryAllocator()
                                .allocateMemory(DataType.FP32, Shape.of(values.length)),
                        DataType.FP32,
                        Layout.rowMajor(Shape.of(values.length)));

        MemoryDomain.copy(heap, src, direct, dst);

        MemoryAccess<ByteBuffer> access = direct.directAccess();
        for (int i = 0; i < values.length; i++) {
            assertEquals(
                    values[i],
                    access.readFloat(dst.memory(), (long) i * Float.BYTES),
                    0.0f,
                    "element " + i);
        }
    }

    // ---- 2. Raw bytes are native order on every domain ----

    @Test
    void segmentBytesAreNativeOrder() {
        MemoryDomain<MemorySegment> seg = MemoryDomains.of(MemoryAllocators.newScopedArena());
        MemoryAccess<MemorySegment> access = seg.directAccess();
        MemoryView<MemorySegment> view =
                MemoryView.of(
                        seg.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(1)),
                        DataType.FP32,
                        Layout.rowMajor(Shape.of(1)));
        access.writeFloat(view.memory(), 0, 1.5f); // 0x3FC00000

        byte[] raw = view.memory().base().toArray(ValueLayout.JAVA_BYTE);
        byte[] expected = bytesOf(Float.floatToIntBits(1.5f));
        assertArrayEquals(expected, raw, "MemorySegment bytes must be native order");
    }

    @Test
    void byteBufferBytesAreNativeOrder() {
        MemoryDomain<ByteBuffer> bb =
                MemoryDomains.ofByteBuffer(MemoryAllocators.newByteBuffer(false));
        MemoryAccess<ByteBuffer> access = bb.directAccess();
        MemoryView<ByteBuffer> view =
                MemoryView.of(
                        bb.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(1)),
                        DataType.FP32,
                        Layout.rowMajor(Shape.of(1)));
        access.writeFloat(view.memory(), 0, 1.5f);

        byte[] raw = new byte[Float.BYTES];
        view.memory().base().get(0, raw);
        assertArrayEquals(
                bytesOf(Float.floatToIntBits(1.5f)), raw, "ByteBuffer bytes must be native order");
    }

    // ---- 3. Same value, different domains, identical raw bytes ----

    @Test
    void sameFloatHasIdenticalBytesInEveryDomain() {
        List<MemoryDomain<?>> domains = f32Domains().toList();
        byte[] reference = null;
        for (MemoryDomain<?> d : domains) {
            byte[] raw = rawFloatBytes(d, 1.5f);
            if (reference == null) {
                reference = raw;
            } else {
                assertArrayEquals(
                        reference,
                        raw,
                        "float bytes differ between "
                                + domains.get(0).device()
                                + " and "
                                + d.device());
            }
        }
        assertArrayEquals(
                bytesOf(Float.floatToIntBits(1.5f)), reference, "reference must be native order");
    }

    // ---- helpers ----

    private static <B> byte[] rawFloatBytes(MemoryDomain<B> domain, float value) {
        MemoryAccess<B> access = domain.directAccess();
        MemoryView<B> view =
                MemoryView.of(
                        domain.memoryAllocator().allocateMemory(DataType.FP32, Shape.of(1)),
                        DataType.FP32,
                        Layout.rowMajor(Shape.of(1)));
        access.writeFloat(view.memory(), 0, value);
        if (view.memory().base() instanceof MemorySegment segment) {
            return segment.toArray(ValueLayout.JAVA_BYTE);
        }
        if (view.memory().base() instanceof ByteBuffer buffer) {
            byte[] raw = new byte[Float.BYTES];
            buffer.get(0, raw);
            return raw;
        }
        if (view.memory().base() instanceof float[] floats) {
            return bytesOf(Float.floatToIntBits(floats[0]));
        }
        if (view.memory().base() instanceof byte[] bytes) {
            return Arrays.copyOf(bytes, Float.BYTES);
        }
        throw new IllegalStateException("unexpected base: " + view.memory().base().getClass());
    }

    private static byte[] bytesOf(int bits) {
        return ByteBuffer.allocate(Integer.BYTES)
                .order(ByteOrder.nativeOrder())
                .putInt(bits)
                .array();
    }

    private static <B> MemoryView<B> allocate(
            MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return MemoryView.of(
                domain.memoryAllocator().allocateMemory(dataType, shape),
                dataType,
                Layout.rowMajor(shape));
    }

    private static <B> MemoryView<B> viewOf(MemoryDomain<B> domain, float[] values) {
        MemoryView<B> view =
                MemoryView.of(
                        domain.memoryAllocator()
                                .allocateMemory(DataType.FP32, Shape.of(values.length)),
                        DataType.FP32,
                        Layout.rowMajor(Shape.of(values.length)));
        MemoryAccess<B> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            access.writeFloat(view.memory(), (long) i * Float.BYTES, values[i]);
        }
        return view;
    }

    private static <B> float[] baseFloats(MemoryDomain<B> domain, MemoryView<B> view) {
        MemoryAccess<B> access = domain.directAccess();
        float[] out = new float[(int) view.shape().size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = access.readFloat(view.memory(), (long) i * Float.BYTES);
        }
        return out;
    }

    private static <B> MemoryView<B> viewOf(MemoryDomain<B> domain, byte[] values) {
        MemoryView<B> view =
                MemoryView.of(
                        domain.memoryAllocator()
                                .allocateMemory(DataType.I8, Shape.of(values.length)),
                        DataType.I8,
                        Layout.rowMajor(Shape.of(values.length)));
        MemoryAccess<B> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            access.writeByte(view.memory(), i, values[i]);
        }
        return view;
    }

    private static <B> MemoryView<B> viewOf(MemoryDomain<B> domain, long[] values) {
        MemoryView<B> view =
                MemoryView.of(
                        domain.memoryAllocator()
                                .allocateMemory(DataType.I64, Shape.of(values.length)),
                        DataType.I64,
                        Layout.rowMajor(Shape.of(values.length)));
        MemoryAccess<B> access = domain.directAccess();
        for (int i = 0; i < values.length; i++) {
            access.writeLong(view.memory(), (long) i * Long.BYTES, values[i]);
        }
        return view;
    }
}

package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.stream.Stream;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests for {@link MemoryOperations#copy(MemoryOperations, Memory, long, MemoryOperations, Memory,
 * long, long)} which auto-allocates an intermediate buffer.
 *
 * <p>The method uses a minimum 4K buffer and dynamically sizes based on transfer size.
 */
class MemoryOperationsAutoBufferTest {

    private static final int MIN_BUFFER_SIZE = 4 << 10; // 4K

    // Native contexts
    @AutoClose MemoryContext<MemorySegment> nativeContext = ContextFactory.ofMemorySegment();

    @AutoClose
    MemoryContext<ByteBuffer> bufferContext =
            ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false));

    @AutoClose
    MemoryContext<ByteBuffer> directBufferContext =
            ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true));

    // Primitive array contexts
    @AutoClose MemoryContext<byte[]> byteContext = ContextFactory.ofBytes();

    @AutoClose MemoryContext<short[]> shortContext = ContextFactory.ofShorts();

    @AutoClose MemoryContext<int[]> intContext = ContextFactory.ofInts();

    @AutoClose MemoryContext<long[]> longContext = ContextFactory.ofLongs();

    @AutoClose MemoryContext<float[]> floatContext = ContextFactory.ofFloats();

    @AutoClose MemoryContext<double[]> doubleContext = ContextFactory.ofDoubles();

    // ========== Zero and small copies ==========

    @Test
    void copyZeroBytes() {
        Memory<float[]> src = floatContext.memoryAllocator().allocateMemory(16);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(16);

        assertDoesNotThrow(
                () ->
                        MemoryOperations.copy(
                                floatContext.memoryOperations(),
                                src,
                                0,
                                nativeContext.memoryOperations(),
                                dst,
                                0,
                                0));
    }

    @Test
    void copySingleByte() {
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(1);
        Memory<ByteBuffer> dst = bufferContext.memoryAllocator().allocateMemory(1);

        nativeContext.memoryAccess().writeByte(src, 0, (byte) 0x42);

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                bufferContext.memoryOperations(),
                dst,
                0,
                1);

        assertEquals((byte) 0x42, bufferContext.memoryAccess().readByte(dst, 0));
    }

    @Test
    void copySingleFloat() {
        Memory<float[]> src = floatContext.memoryAllocator().allocateMemory(Float.BYTES);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(Float.BYTES);

        floatContext.memoryAccess().writeFloat(src, 0, 3.14159f);

        MemoryOperations.copy(
                floatContext.memoryOperations(),
                src,
                0,
                nativeContext.memoryOperations(),
                dst,
                0,
                Float.BYTES);

        assertEquals(3.14159f, nativeContext.memoryAccess().readFloat(dst, 0), 0.00001f);
    }

    // ========== Copies smaller than minimum buffer (4K) ==========

    @ParameterizedTest
    @ValueSource(ints = {64, 256, 512, 1024, 2048, 4095})
    void copyBelowMinBufferSize(int byteSize) {
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferContext.memoryAllocator().allocateMemory(byteSize);

        // Write pattern to source
        int floatCount = byteSize / Float.BYTES;
        for (int i = 0; i < floatCount; i++) {
            nativeContext.memoryAccess().writeFloat(src, (long) i * Float.BYTES, i * 1.5f);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                directBufferContext.memoryOperations(),
                dst,
                0,
                byteSize);

        // Verify pattern
        for (int i = 0; i < floatCount; i++) {
            assertEquals(
                    i * 1.5f,
                    directBufferContext.memoryAccess().readFloat(dst, (long) i * Float.BYTES),
                    0.00001f,
                    "Mismatch at float index " + i);
        }
    }

    // ========== Copies at exactly 4K boundary ==========

    @Test
    void copyExactly4K() {
        int byteSize = MIN_BUFFER_SIZE;
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<float[]> dst = floatContext.memoryAllocator().allocateMemory(byteSize);

        int floatCount = byteSize / Float.BYTES;
        for (int i = 0; i < floatCount; i++) {
            nativeContext.memoryAccess().writeFloat(src, (long) i * Float.BYTES, i + 0.25f);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                floatContext.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < floatCount; i++) {
            assertEquals(
                    i + 0.25f,
                    floatContext.memoryAccess().readFloat(dst, (long) i * Float.BYTES),
                    0.00001f);
        }
    }

    // ========== Copies larger than 4K (require chunking) ==========

    @ParameterizedTest
    @ValueSource(ints = {4097, 8192, 16384, 32768, 65536})
    void copyAboveMinBufferSize(int byteSize) {
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferContext.memoryAllocator().allocateMemory(byteSize);

        // Write distinct pattern
        int intCount = byteSize / Integer.BYTES;
        for (int i = 0; i < intCount; i++) {
            nativeContext.memoryAccess().writeInt(src, (long) i * Integer.BYTES, i * 7 + 13);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                directBufferContext.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < intCount; i++) {
            assertEquals(
                    i * 7 + 13,
                    directBufferContext.memoryAccess().readInt(dst, (long) i * Integer.BYTES),
                    "Mismatch at int index " + i);
        }
    }

    // ========== Large copies (1MB+) ==========

    @Test
    void copyOneMegabyte() {
        int byteSize = 1 << 20; // 1 MB
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferContext.memoryAllocator().allocateMemory(byteSize);

        // Write pattern at boundaries and middle
        long[] checkpoints = {0, 1000, byteSize / 2, byteSize - Long.BYTES};
        for (int i = 0; i < checkpoints.length; i++) {
            nativeContext.memoryAccess().writeLong(src, checkpoints[i], 0xDEADBEEF_CAFEBABEL + i);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                directBufferContext.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < checkpoints.length; i++) {
            assertEquals(
                    0xDEADBEEF_CAFEBABEL + i,
                    directBufferContext.memoryAccess().readLong(dst, checkpoints[i]),
                    "Mismatch at checkpoint " + i);
        }
    }

    @Test
    void copyFourMegabytes() {
        int byteSize = 4 << 20; // 4 MB
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(byteSize);

        // Write sequential longs
        int longCount = byteSize / Long.BYTES;
        for (int i = 0; i < longCount; i += 1000) { // Sample every 1000th
            nativeContext.memoryAccess().writeLong(src, (long) i * Long.BYTES, i);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                nativeContext.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < longCount; i += 1000) {
            assertEquals(
                    i,
                    nativeContext.memoryAccess().readLong(dst, (long) i * Long.BYTES),
                    "Mismatch at long index " + i);
        }
    }

    // ========== Copies with offsets ==========

    @Test
    void copyWithSourceOffset() {
        int byteSize = 8192;
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<float[]> dst = floatContext.memoryAllocator().allocateMemory(byteSize / 2);

        // Write to second half of source
        int floatCount = byteSize / 2 / Float.BYTES;
        for (int i = 0; i < floatCount; i++) {
            nativeContext
                    .memoryAccess()
                    .writeFloat(src, byteSize / 2 + (long) i * Float.BYTES, i * 2.5f);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                byteSize / 2, // source offset
                floatContext.memoryOperations(),
                dst,
                0,
                byteSize / 2);

        for (int i = 0; i < floatCount; i++) {
            assertEquals(
                    i * 2.5f,
                    floatContext.memoryAccess().readFloat(dst, (long) i * Float.BYTES),
                    0.00001f);
        }
    }

    @Test
    void copyWithDestOffset() {
        int byteSize = 8192;
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize / 2);
        Memory<ByteBuffer> dst = directBufferContext.memoryAllocator().allocateMemory(byteSize);

        int intCount = byteSize / 2 / Integer.BYTES;
        for (int i = 0; i < intCount; i++) {
            nativeContext.memoryAccess().writeInt(src, (long) i * Integer.BYTES, i + 100);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                directBufferContext.memoryOperations(),
                dst,
                byteSize / 2, // dest offset
                byteSize / 2);

        for (int i = 0; i < intCount; i++) {
            assertEquals(
                    i + 100,
                    directBufferContext
                            .memoryAccess()
                            .readInt(dst, byteSize / 2 + (long) i * Integer.BYTES));
        }
    }

    @Test
    void copyWithBothOffsets() {
        int totalSize = 16384;
        int copySize = 4096;
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(totalSize);
        Memory<ByteBuffer> dst = directBufferContext.memoryAllocator().allocateMemory(totalSize);

        // Write to middle of source
        int srcOffset = 4096;
        int dstOffset = 8192;
        int longCount = copySize / Long.BYTES;
        for (int i = 0; i < longCount; i++) {
            nativeContext.memoryAccess().writeLong(src, srcOffset + (long) i * Long.BYTES, i * 17L);
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                srcOffset,
                directBufferContext.memoryOperations(),
                dst,
                dstOffset,
                copySize);

        for (int i = 0; i < longCount; i++) {
            assertEquals(
                    i * 17L,
                    directBufferContext
                            .memoryAccess()
                            .readLong(dst, dstOffset + (long) i * Long.BYTES));
        }
    }

    // ========== Cross-context combinations ==========

    static Stream<Arguments> contextPairs() {
        // Only include pairs where we can set up and verify data using byte-capable contexts
        return Stream.of(
                // byte[] <-> other byte-capable contexts
                Arguments.of("byte[] -> MemorySegment", "bytes", "native"),
                Arguments.of("MemorySegment -> byte[]", "native", "bytes"),
                Arguments.of("byte[] -> ByteBuffer", "bytes", "buffer"),
                Arguments.of("ByteBuffer -> byte[]", "buffer", "bytes"),
                Arguments.of("byte[] -> DirectByteBuffer", "bytes", "direct"),
                Arguments.of("DirectByteBuffer -> byte[]", "direct", "bytes"),
                // MemorySegment <-> ByteBuffer variants
                Arguments.of("ByteBuffer -> MemorySegment", "buffer", "native"),
                Arguments.of("MemorySegment -> ByteBuffer", "native", "buffer"),
                Arguments.of("DirectByteBuffer -> MemorySegment", "direct", "native"),
                Arguments.of("MemorySegment -> DirectByteBuffer", "native", "direct"),
                // ByteBuffer variants
                Arguments.of("ByteBuffer -> DirectByteBuffer", "buffer", "direct"),
                Arguments.of("DirectByteBuffer -> ByteBuffer", "direct", "buffer"));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("contextPairs")
    void copyBetweenContextPairs(String name, String srcType, String dstType) {
        int byteSize = 8192; // 8K - requires chunking with 4K buffer

        MemoryContext<?> srcCtx = getContext(srcType);
        MemoryContext<?> dstCtx = getContext(dstType);

        Memory<?> src = srcCtx.memoryAllocator().allocateMemory(byteSize);
        Memory<?> dst = dstCtx.memoryAllocator().allocateMemory(byteSize);

        // Write test pattern using bytes
        for (int i = 0; i < byteSize; i += 64) {
            writeByte(srcCtx, src, i, (byte) ((i / 64) & 0xFF));
        }

        copyUntyped(srcCtx, src, 0, dstCtx, dst, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64) & 0xFF),
                    readByte(dstCtx, dst, i),
                    "Mismatch at offset " + i);
        }
    }

    // ========== Primitive array context tests (use native context for setup/verify) ==========

    static Stream<Arguments> primitiveContexts() {
        return Stream.of(
                Arguments.of("short[]", "shorts"),
                Arguments.of("int[]", "ints"),
                Arguments.of("long[]", "longs"),
                Arguments.of("float[]", "floats"),
                Arguments.of("double[]", "doubles"));
    }

    @ParameterizedTest(name = "MemorySegment -> {0}")
    @MethodSource("primitiveContexts")
    void copyFromNativeToPrimitiveContext(String name, String type) {
        int byteSize = 8192;

        MemoryContext<?> primitiveCtx = getContext(type);
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<?> dst = primitiveCtx.memoryAllocator().allocateMemory(byteSize);

        // Write pattern to native
        for (int i = 0; i < byteSize; i += 64) {
            nativeContext.memoryAccess().writeByte(src, i, (byte) ((i / 64) & 0xFF));
        }

        // Copy native -> primitive
        copyUntyped(nativeContext, src, 0, primitiveCtx, dst, 0, byteSize);

        // Copy primitive -> new native buffer to verify
        Memory<MemorySegment> verify = nativeContext.memoryAllocator().allocateMemory(byteSize);
        copyUntyped(primitiveCtx, dst, 0, nativeContext, verify, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64) & 0xFF),
                    nativeContext.memoryAccess().readByte(verify, i),
                    "Mismatch at offset " + i);
        }
    }

    @ParameterizedTest(name = "{0} -> MemorySegment")
    @MethodSource("primitiveContexts")
    void copyFromPrimitiveContextToNative(String name, String type) {
        int byteSize = 8192;

        MemoryContext<?> primitiveCtx = getContext(type);

        // Write to native first, copy to primitive, then copy back and verify
        Memory<MemorySegment> initial = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<?> primitive = primitiveCtx.memoryAllocator().allocateMemory(byteSize);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(byteSize);

        // Write pattern
        for (int i = 0; i < byteSize; i += 64) {
            nativeContext.memoryAccess().writeByte(initial, i, (byte) ((i / 64 + 50) & 0xFF));
        }

        // Copy native -> primitive -> native
        copyUntyped(nativeContext, initial, 0, primitiveCtx, primitive, 0, byteSize);
        copyUntyped(primitiveCtx, primitive, 0, nativeContext, dst, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64 + 50) & 0xFF),
                    nativeContext.memoryAccess().readByte(dst, i),
                    "Mismatch at offset " + i);
        }
    }

    @ParameterizedTest(name = "{0} -> {0} (same type)")
    @MethodSource("primitiveContexts")
    void copyWithinSamePrimitiveContextType(String name, String type) {
        int byteSize = 8192;

        MemoryContext<?> ctx = getContext(type);
        Memory<?> src = ctx.memoryAllocator().allocateMemory(byteSize);
        Memory<?> dst = ctx.memoryAllocator().allocateMemory(byteSize);

        // Set up via native
        Memory<MemorySegment> setup = nativeContext.memoryAllocator().allocateMemory(byteSize);
        for (int i = 0; i < byteSize; i += 64) {
            nativeContext.memoryAccess().writeByte(setup, i, (byte) ((i / 64 + 100) & 0xFF));
        }
        copyUntyped(nativeContext, setup, 0, ctx, src, 0, byteSize);

        // Copy within same context type
        copyUntyped(ctx, src, 0, ctx, dst, 0, byteSize);

        // Verify via native
        Memory<MemorySegment> verify = nativeContext.memoryAllocator().allocateMemory(byteSize);
        copyUntyped(ctx, dst, 0, nativeContext, verify, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64 + 100) & 0xFF),
                    nativeContext.memoryAccess().readByte(verify, i),
                    "Mismatch at offset " + i);
        }
    }

    // ========== Edge cases ==========

    @Test
    void copyNonAlignedSize() {
        // Copy 4099 bytes - just over 4K, not aligned
        int byteSize = 4099;
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferContext.memoryAllocator().allocateMemory(byteSize);

        // Write bytes at start and end
        nativeContext.memoryAccess().writeByte(src, 0, (byte) 0xAB);
        nativeContext.memoryAccess().writeByte(src, byteSize - 1, (byte) 0xCD);

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                directBufferContext.memoryOperations(),
                dst,
                0,
                byteSize);

        assertEquals((byte) 0xAB, directBufferContext.memoryAccess().readByte(dst, 0));
        assertEquals((byte) 0xCD, directBufferContext.memoryAccess().readByte(dst, byteSize - 1));
    }

    @Test
    void copyPrimeNumberBytes() {
        // 7919 is prime - tests odd chunking
        int byteSize = 7919;
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = bufferContext.memoryAllocator().allocateMemory(byteSize);

        // Write pattern
        for (int i = 0; i < byteSize; i += 100) {
            nativeContext.memoryAccess().writeByte(src, i, (byte) (i % 256));
        }

        MemoryOperations.copy(
                nativeContext.memoryOperations(),
                src,
                0,
                bufferContext.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < byteSize; i += 100) {
            assertEquals(
                    (byte) (i % 256), bufferContext.memoryAccess().readByte(dst, i), "At offset " + i);
        }
    }

    // ========== Granularity violation tests ==========

    @Test
    void copyFailsWhenSourceOffsetNotAlignedToGranularity() {
        // int[] has 4-byte granularity, offset 2 is not aligned
        Memory<int[]> src = intContext.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        intContext.memoryOperations(),
                                        src,
                                        2, // not aligned to 4
                                        nativeContext.memoryOperations(),
                                        dst,
                                        0,
                                        16));

        assertTrue(ex.getMessage().contains("Source offset"));
        assertTrue(ex.getMessage().contains("not aligned"));
    }

    @Test
    void copyFailsWhenDestOffsetNotAlignedToGranularity() {
        // long[] has 8-byte granularity, offset 4 is not aligned
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(32);
        Memory<long[]> dst = longContext.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        nativeContext.memoryOperations(),
                                        src,
                                        0,
                                        longContext.memoryOperations(),
                                        dst,
                                        4, // not aligned to 8
                                        16));

        assertTrue(ex.getMessage().contains("Destination offset"));
        assertTrue(ex.getMessage().contains("not aligned"));
    }

    @Test
    void copyFailsWhenByteSizeNotMultipleOfSourceGranularity() {
        // float[] has 4-byte granularity, size 10 is not a multiple of 4
        Memory<float[]> src = floatContext.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        floatContext.memoryOperations(),
                                        src,
                                        0,
                                        nativeContext.memoryOperations(),
                                        dst,
                                        0,
                                        10)); // not a multiple of 4

        assertTrue(ex.getMessage().contains("Byte size"));
        assertTrue(ex.getMessage().contains("source granularity"));
    }

    @Test
    void copyFailsWhenByteSizeNotMultipleOfDestGranularity() {
        // int[] has 4-byte granularity, double[] has 8-byte granularity
        // size 12 is multiple of 4 but not of 8
        Memory<int[]> src = intContext.memoryAllocator().allocateMemory(32);
        Memory<double[]> dst = doubleContext.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        intContext.memoryOperations(),
                                        src,
                                        0,
                                        doubleContext.memoryOperations(),
                                        dst,
                                        0,
                                        12)); // multiple of 4 but not 8

        assertTrue(ex.getMessage().contains("Byte size"));
        assertTrue(ex.getMessage().contains("destination granularity"));
    }

    @Test
    void copySucceedsWithAlignedOffsetsAndSize() {
        // All aligned: int[] (4-byte granularity), offset 8, size 16
        Memory<int[]> src = intContext.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(32);

        assertDoesNotThrow(
                () ->
                        MemoryOperations.copy(
                                intContext.memoryOperations(),
                                src,
                                8, // aligned to 4
                                nativeContext.memoryOperations(),
                                dst,
                                4, // aligned to 1 (native)
                                16)); // multiple of 4
    }

    @Test
    void copyBetweenLargeGranularityContextsWithProperAlignment() {
        // long[] (8-byte) to double[] (8-byte): offset and size must be multiples of 8
        Memory<long[]> src = longContext.memoryAllocator().allocateMemory(64);
        Memory<double[]> dst = doubleContext.memoryAllocator().allocateMemory(64);

        assertDoesNotThrow(
                () ->
                        MemoryOperations.copy(
                                longContext.memoryOperations(),
                                src,
                                16, // multiple of 8
                                doubleContext.memoryOperations(),
                                dst,
                                8, // multiple of 8
                                24)); // multiple of 8
    }

    @Test
    void copyFailsWithOddSourceOffsetOnShortArray() {
        // short[] has 2-byte granularity
        Memory<short[]> src = shortContext.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        shortContext.memoryOperations(),
                                        src,
                                        3, // not aligned to 2
                                        nativeContext.memoryOperations(),
                                        dst,
                                        0,
                                        16));

        assertTrue(ex.getMessage().contains("Source offset"));
    }

    @Test
    void copyFailsWithOddByteSizeToShortArray() {
        // short[] has 2-byte granularity, size 15 is odd
        Memory<MemorySegment> src = nativeContext.memoryAllocator().allocateMemory(32);
        Memory<short[]> dst = shortContext.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        nativeContext.memoryOperations(),
                                        src,
                                        0,
                                        shortContext.memoryOperations(),
                                        dst,
                                        0,
                                        15)); // odd, not multiple of 2

        assertTrue(ex.getMessage().contains("Byte size"));
    }

    @Test
    void copyWithBufferAlsoRespectsGranularity() {
        // Test the overload that takes an explicit buffer
        Memory<long[]> src = longContext.memoryAllocator().allocateMemory(64);
        Memory<double[]> dst = doubleContext.memoryAllocator().allocateMemory(64);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4096);

        // Should fail: size 12 is not multiple of 8
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        longContext.memoryOperations(),
                                        src,
                                        0,
                                        doubleContext.memoryOperations(),
                                        dst,
                                        0,
                                        12, // not multiple of 8
                                        buffer));

        assertTrue(ex.getMessage().contains("Byte size"));
    }

    // ========== Helpers ==========

    private MemoryContext<?> getContext(String type) {
        return switch (type) {
            case "bytes" -> byteContext;
            case "shorts" -> shortContext;
            case "ints" -> intContext;
            case "longs" -> longContext;
            case "floats" -> floatContext;
            case "doubles" -> doubleContext;
            case "native" -> nativeContext;
            case "buffer" -> bufferContext;
            case "direct" -> directBufferContext;
            default -> throw new IllegalArgumentException("Unknown context: " + type);
        };
    }

    @SuppressWarnings("unchecked")
    private static <B> void writeByte(MemoryContext<B> ctx, Memory<?> mem, long offset, byte v) {
        ctx.memoryAccess().writeByte((Memory<B>) mem, offset, v);
    }

    @SuppressWarnings("unchecked")
    private static <B> byte readByte(MemoryContext<B> ctx, Memory<?> mem, long offset) {
        return ctx.memoryAccess().readByte((Memory<B>) mem, offset);
    }

    @SuppressWarnings("unchecked")
    private static <S, D> void copyUntyped(
            MemoryContext<S> srcCtx,
            Memory<?> src,
            long srcOffset,
            MemoryContext<D> dstCtx,
            Memory<?> dst,
            long dstOffset,
            long byteSize) {
        MemoryOperations.copy(
                srcCtx.memoryOperations(),
                (Memory<S>) src,
                srcOffset,
                dstCtx.memoryOperations(),
                (Memory<D>) dst,
                dstOffset,
                byteSize);
    }
}

package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
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

    // Native domains
    @AutoClose MemoryDomain<MemorySegment> nativeDomain = DomainFactory.ofMemorySegment();

    @AutoClose
    MemoryDomain<ByteBuffer> bufferDomain =
            DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false));

    @AutoClose
    MemoryDomain<ByteBuffer> directBufferDomain =
            DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true));

    // Primitive array domains
    @AutoClose MemoryDomain<byte[]> byteDomain = DomainFactory.ofBytes();

    @AutoClose MemoryDomain<short[]> shortDomain = DomainFactory.ofShorts();

    @AutoClose MemoryDomain<int[]> intDomain = DomainFactory.ofInts();

    @AutoClose MemoryDomain<long[]> longDomain = DomainFactory.ofLongs();

    @AutoClose MemoryDomain<float[]> floatDomain = DomainFactory.ofFloats();

    @AutoClose MemoryDomain<double[]> doubleDomain = DomainFactory.ofDoubles();

    // ========== Zero and small copies ==========

    @Test
    void copyZeroBytes() {
        Memory<float[]> src = floatDomain.memoryAllocator().allocateMemory(16);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(16);

        assertDoesNotThrow(
                () ->
                        MemoryOperations.copy(
                                floatDomain.memoryOperations(),
                                src,
                                0,
                                nativeDomain.memoryOperations(),
                                dst,
                                0,
                                0));
    }

    @Test
    void copySingleByte() {
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(1);
        Memory<ByteBuffer> dst = bufferDomain.memoryAllocator().allocateMemory(1);

        nativeDomain.directAccess().writeByte(src, 0, (byte) 0x42);

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                bufferDomain.memoryOperations(),
                dst,
                0,
                1);

        assertEquals((byte) 0x42, bufferDomain.directAccess().readByte(dst, 0));
    }

    @Test
    void copySingleFloat() {
        Memory<float[]> src = floatDomain.memoryAllocator().allocateMemory(Float.BYTES);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(Float.BYTES);

        floatDomain.directAccess().writeFloat(src, 0, 3.14159f);

        MemoryOperations.copy(
                floatDomain.memoryOperations(),
                src,
                0,
                nativeDomain.memoryOperations(),
                dst,
                0,
                Float.BYTES);

        assertEquals(3.14159f, nativeDomain.directAccess().readFloat(dst, 0), 0.00001f);
    }

    // ========== Copies smaller than minimum buffer (4K) ==========

    @ParameterizedTest
    @ValueSource(ints = {64, 256, 512, 1024, 2048, 4095})
    void copyBelowMinBufferSize(int byteSize) {
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferDomain.memoryAllocator().allocateMemory(byteSize);

        // Write pattern to source
        int floatCount = byteSize / Float.BYTES;
        for (int i = 0; i < floatCount; i++) {
            nativeDomain.directAccess().writeFloat(src, (long) i * Float.BYTES, i * 1.5f);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                directBufferDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        // Verify pattern
        for (int i = 0; i < floatCount; i++) {
            assertEquals(
                    i * 1.5f,
                    directBufferDomain.directAccess().readFloat(dst, (long) i * Float.BYTES),
                    0.00001f,
                    "Mismatch at float index " + i);
        }
    }

    // ========== Copies at exactly 4K boundary ==========

    @Test
    void copyExactly4K() {
        int byteSize = MIN_BUFFER_SIZE;
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<float[]> dst = floatDomain.memoryAllocator().allocateMemory(byteSize);

        int floatCount = byteSize / Float.BYTES;
        for (int i = 0; i < floatCount; i++) {
            nativeDomain.directAccess().writeFloat(src, (long) i * Float.BYTES, i + 0.25f);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                floatDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < floatCount; i++) {
            assertEquals(
                    i + 0.25f,
                    floatDomain.directAccess().readFloat(dst, (long) i * Float.BYTES),
                    0.00001f);
        }
    }

    // ========== Copies larger than 4K (require chunking) ==========

    @ParameterizedTest
    @ValueSource(ints = {4097, 8192, 16384, 32768, 65536})
    void copyAboveMinBufferSize(int byteSize) {
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferDomain.memoryAllocator().allocateMemory(byteSize);

        // Write distinct pattern
        int intCount = byteSize / Integer.BYTES;
        for (int i = 0; i < intCount; i++) {
            nativeDomain.directAccess().writeInt(src, (long) i * Integer.BYTES, i * 7 + 13);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                directBufferDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < intCount; i++) {
            assertEquals(
                    i * 7 + 13,
                    directBufferDomain.directAccess().readInt(dst, (long) i * Integer.BYTES),
                    "Mismatch at int index " + i);
        }
    }

    // ========== Large copies (1MB+) ==========

    @Test
    void copyOneMegabyte() {
        int byteSize = 1 << 20; // 1 MB
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferDomain.memoryAllocator().allocateMemory(byteSize);

        // Write pattern at boundaries and middle
        long[] checkpoints = {0, 1000, byteSize / 2, byteSize - Long.BYTES};
        for (int i = 0; i < checkpoints.length; i++) {
            nativeDomain.directAccess().writeLong(src, checkpoints[i], 0xDEADBEEF_CAFEBABEL + i);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                directBufferDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < checkpoints.length; i++) {
            assertEquals(
                    0xDEADBEEF_CAFEBABEL + i,
                    directBufferDomain.directAccess().readLong(dst, checkpoints[i]),
                    "Mismatch at checkpoint " + i);
        }
    }

    @Test
    void copyFourMegabytes() {
        int byteSize = 4 << 20; // 4 MB
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(byteSize);

        // Write sequential longs
        int longCount = byteSize / Long.BYTES;
        for (int i = 0; i < longCount; i += 1000) { // Sample every 1000th
            nativeDomain.directAccess().writeLong(src, (long) i * Long.BYTES, i);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                nativeDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < longCount; i += 1000) {
            assertEquals(
                    i,
                    nativeDomain.directAccess().readLong(dst, (long) i * Long.BYTES),
                    "Mismatch at long index " + i);
        }
    }

    // ========== Copies with offsets ==========

    @Test
    void copyWithSourceOffset() {
        int byteSize = 8192;
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<float[]> dst = floatDomain.memoryAllocator().allocateMemory(byteSize / 2);

        // Write to second half of source
        int floatCount = byteSize / 2 / Float.BYTES;
        for (int i = 0; i < floatCount; i++) {
            nativeDomain
                    .directAccess()
                    .writeFloat(src, byteSize / 2 + (long) i * Float.BYTES, i * 2.5f);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                byteSize / 2, // source offset
                floatDomain.memoryOperations(),
                dst,
                0,
                byteSize / 2);

        for (int i = 0; i < floatCount; i++) {
            assertEquals(
                    i * 2.5f,
                    floatDomain.directAccess().readFloat(dst, (long) i * Float.BYTES),
                    0.00001f);
        }
    }

    @Test
    void copyWithDestOffset() {
        int byteSize = 8192;
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize / 2);
        Memory<ByteBuffer> dst = directBufferDomain.memoryAllocator().allocateMemory(byteSize);

        int intCount = byteSize / 2 / Integer.BYTES;
        for (int i = 0; i < intCount; i++) {
            nativeDomain.directAccess().writeInt(src, (long) i * Integer.BYTES, i + 100);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                directBufferDomain.memoryOperations(),
                dst,
                byteSize / 2, // dest offset
                byteSize / 2);

        for (int i = 0; i < intCount; i++) {
            assertEquals(
                    i + 100,
                    directBufferDomain
                            .directAccess()
                            .readInt(dst, byteSize / 2 + (long) i * Integer.BYTES));
        }
    }

    @Test
    void copyWithBothOffsets() {
        int totalSize = 16384;
        int copySize = 4096;
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(totalSize);
        Memory<ByteBuffer> dst = directBufferDomain.memoryAllocator().allocateMemory(totalSize);

        // Write to middle of source
        int srcOffset = 4096;
        int dstOffset = 8192;
        int longCount = copySize / Long.BYTES;
        for (int i = 0; i < longCount; i++) {
            nativeDomain.directAccess().writeLong(src, srcOffset + (long) i * Long.BYTES, i * 17L);
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                srcOffset,
                directBufferDomain.memoryOperations(),
                dst,
                dstOffset,
                copySize);

        for (int i = 0; i < longCount; i++) {
            assertEquals(
                    i * 17L,
                    directBufferDomain
                            .directAccess()
                            .readLong(dst, dstOffset + (long) i * Long.BYTES));
        }
    }

    // ========== Cross-domain combinations ==========

    static Stream<Arguments> domainPairs() {
        // Only include pairs where we can set up and verify data using byte-capable domains
        return Stream.of(
                // byte[] <-> other byte-capable domains
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
    @MethodSource("domainPairs")
    void copyBetweenDomainPairs(String name, String srcType, String dstType) {
        int byteSize = 8192; // 8K - requires chunking with 4K buffer

        MemoryDomain<?> srcCtx = getDomain(srcType);
        MemoryDomain<?> dstCtx = getDomain(dstType);

        Memory<?> src = srcCtx.memoryAllocator().allocateMemory(byteSize);
        Memory<?> dst = dstCtx.memoryAllocator().allocateMemory(byteSize);

        // Write test pattern using bytes
        for (int i = 0; i < byteSize; i += 64) {
            writeByte(srcCtx, src, i, (byte) ((i / 64) & 0xFF));
        }

        copyUntyped(srcCtx, src, 0, dstCtx, dst, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64) & 0xFF), readByte(dstCtx, dst, i), "Mismatch at offset " + i);
        }
    }

    // ========== Primitive array domain tests (use native domain for setup/verify) ==========

    static Stream<Arguments> primitiveDomains() {
        return Stream.of(
                Arguments.of("short[]", "shorts"),
                Arguments.of("int[]", "ints"),
                Arguments.of("long[]", "longs"),
                Arguments.of("float[]", "floats"),
                Arguments.of("double[]", "doubles"));
    }

    @ParameterizedTest(name = "MemorySegment -> {0}")
    @MethodSource("primitiveDomains")
    void copyFromNativeToPrimitiveDomain(String name, String type) {
        int byteSize = 8192;

        MemoryDomain<?> primitiveCtx = getDomain(type);
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<?> dst = primitiveCtx.memoryAllocator().allocateMemory(byteSize);

        // Write pattern to native
        for (int i = 0; i < byteSize; i += 64) {
            nativeDomain.directAccess().writeByte(src, i, (byte) ((i / 64) & 0xFF));
        }

        // Copy native -> primitive
        copyUntyped(nativeDomain, src, 0, primitiveCtx, dst, 0, byteSize);

        // Copy primitive -> new native buffer to verify
        Memory<MemorySegment> verify = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        copyUntyped(primitiveCtx, dst, 0, nativeDomain, verify, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64) & 0xFF),
                    nativeDomain.directAccess().readByte(verify, i),
                    "Mismatch at offset " + i);
        }
    }

    @ParameterizedTest(name = "{0} -> MemorySegment")
    @MethodSource("primitiveDomains")
    void copyFromPrimitiveDomainToNative(String name, String type) {
        int byteSize = 8192;

        MemoryDomain<?> primitiveCtx = getDomain(type);

        // Write to native first, copy to primitive, then copy back and verify
        Memory<MemorySegment> initial = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<?> primitive = primitiveCtx.memoryAllocator().allocateMemory(byteSize);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(byteSize);

        // Write pattern
        for (int i = 0; i < byteSize; i += 64) {
            nativeDomain.directAccess().writeByte(initial, i, (byte) ((i / 64 + 50) & 0xFF));
        }

        // Copy native -> primitive -> native
        copyUntyped(nativeDomain, initial, 0, primitiveCtx, primitive, 0, byteSize);
        copyUntyped(primitiveCtx, primitive, 0, nativeDomain, dst, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64 + 50) & 0xFF),
                    nativeDomain.directAccess().readByte(dst, i),
                    "Mismatch at offset " + i);
        }
    }

    @ParameterizedTest(name = "{0} -> {0} (same type)")
    @MethodSource("primitiveDomains")
    void copyWithinSamePrimitiveDomainType(String name, String type) {
        int byteSize = 8192;

        MemoryDomain<?> ctx = getDomain(type);
        Memory<?> src = ctx.memoryAllocator().allocateMemory(byteSize);
        Memory<?> dst = ctx.memoryAllocator().allocateMemory(byteSize);

        // Set up via native
        Memory<MemorySegment> setup = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        for (int i = 0; i < byteSize; i += 64) {
            nativeDomain.directAccess().writeByte(setup, i, (byte) ((i / 64 + 100) & 0xFF));
        }
        copyUntyped(nativeDomain, setup, 0, ctx, src, 0, byteSize);

        // Copy within same domain type
        copyUntyped(ctx, src, 0, ctx, dst, 0, byteSize);

        // Verify via native
        Memory<MemorySegment> verify = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        copyUntyped(ctx, dst, 0, nativeDomain, verify, 0, byteSize);

        for (int i = 0; i < byteSize; i += 64) {
            assertEquals(
                    (byte) ((i / 64 + 100) & 0xFF),
                    nativeDomain.directAccess().readByte(verify, i),
                    "Mismatch at offset " + i);
        }
    }

    // ========== Edge cases ==========

    @Test
    void copyNonAlignedSize() {
        // Copy 4099 bytes - just over 4K, not aligned
        int byteSize = 4099;
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = directBufferDomain.memoryAllocator().allocateMemory(byteSize);

        // Write bytes at start and end
        nativeDomain.directAccess().writeByte(src, 0, (byte) 0xAB);
        nativeDomain.directAccess().writeByte(src, byteSize - 1, (byte) 0xCD);

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                directBufferDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        assertEquals((byte) 0xAB, directBufferDomain.directAccess().readByte(dst, 0));
        assertEquals((byte) 0xCD, directBufferDomain.directAccess().readByte(dst, byteSize - 1));
    }

    @Test
    void copyPrimeNumberBytes() {
        // 7919 is prime - tests odd chunking
        int byteSize = 7919;
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(byteSize);
        Memory<ByteBuffer> dst = bufferDomain.memoryAllocator().allocateMemory(byteSize);

        // Write pattern
        for (int i = 0; i < byteSize; i += 100) {
            nativeDomain.directAccess().writeByte(src, i, (byte) (i % 256));
        }

        MemoryOperations.copy(
                nativeDomain.memoryOperations(),
                src,
                0,
                bufferDomain.memoryOperations(),
                dst,
                0,
                byteSize);

        for (int i = 0; i < byteSize; i += 100) {
            assertEquals(
                    (byte) (i % 256),
                    bufferDomain.directAccess().readByte(dst, i),
                    "At offset " + i);
        }
    }

    // ========== Granularity violation tests ==========

    @Test
    void copyFailsWhenSourceOffsetNotAlignedToGranularity() {
        // int[] has 4-byte granularity, offset 2 is not aligned
        Memory<int[]> src = intDomain.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        intDomain.memoryOperations(),
                                        src,
                                        2, // not aligned to 4
                                        nativeDomain.memoryOperations(),
                                        dst,
                                        0,
                                        16));

        assertTrue(ex.getMessage().contains("Source offset"));
        assertTrue(ex.getMessage().contains("not aligned"));
    }

    @Test
    void copyFailsWhenDestOffsetNotAlignedToGranularity() {
        // long[] has 8-byte granularity, offset 4 is not aligned
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(32);
        Memory<long[]> dst = longDomain.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        nativeDomain.memoryOperations(),
                                        src,
                                        0,
                                        longDomain.memoryOperations(),
                                        dst,
                                        4, // not aligned to 8
                                        16));

        assertTrue(ex.getMessage().contains("Destination offset"));
        assertTrue(ex.getMessage().contains("not aligned"));
    }

    @Test
    void copyFailsWhenByteSizeNotMultipleOfSourceGranularity() {
        // float[] has 4-byte granularity, size 10 is not a multiple of 4
        Memory<float[]> src = floatDomain.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        floatDomain.memoryOperations(),
                                        src,
                                        0,
                                        nativeDomain.memoryOperations(),
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
        Memory<int[]> src = intDomain.memoryAllocator().allocateMemory(32);
        Memory<double[]> dst = doubleDomain.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        intDomain.memoryOperations(),
                                        src,
                                        0,
                                        doubleDomain.memoryOperations(),
                                        dst,
                                        0,
                                        12)); // multiple of 4 but not 8

        assertTrue(ex.getMessage().contains("Byte size"));
        assertTrue(ex.getMessage().contains("destination granularity"));
    }

    @Test
    void copySucceedsWithAlignedOffsetsAndSize() {
        // All aligned: int[] (4-byte granularity), offset 8, size 16
        Memory<int[]> src = intDomain.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(32);

        assertDoesNotThrow(
                () ->
                        MemoryOperations.copy(
                                intDomain.memoryOperations(),
                                src,
                                8, // aligned to 4
                                nativeDomain.memoryOperations(),
                                dst,
                                4, // aligned to 1 (native)
                                16)); // multiple of 4
    }

    @Test
    void copyBetweenLargeGranularityDomainsWithProperAlignment() {
        // long[] (8-byte) to double[] (8-byte): offset and size must be multiples of 8
        Memory<long[]> src = longDomain.memoryAllocator().allocateMemory(64);
        Memory<double[]> dst = doubleDomain.memoryAllocator().allocateMemory(64);

        assertDoesNotThrow(
                () ->
                        MemoryOperations.copy(
                                longDomain.memoryOperations(),
                                src,
                                16, // multiple of 8
                                doubleDomain.memoryOperations(),
                                dst,
                                8, // multiple of 8
                                24)); // multiple of 8
    }

    @Test
    void copyFailsWithOddSourceOffsetOnShortArray() {
        // short[] has 2-byte granularity
        Memory<short[]> src = shortDomain.memoryAllocator().allocateMemory(32);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        shortDomain.memoryOperations(),
                                        src,
                                        3, // not aligned to 2
                                        nativeDomain.memoryOperations(),
                                        dst,
                                        0,
                                        16));

        assertTrue(ex.getMessage().contains("Source offset"));
    }

    @Test
    void copyFailsWithOddByteSizeToShortArray() {
        // short[] has 2-byte granularity, size 15 is odd
        Memory<MemorySegment> src = nativeDomain.memoryAllocator().allocateMemory(32);
        Memory<short[]> dst = shortDomain.memoryAllocator().allocateMemory(32);

        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        nativeDomain.memoryOperations(),
                                        src,
                                        0,
                                        shortDomain.memoryOperations(),
                                        dst,
                                        0,
                                        15)); // odd, not multiple of 2

        assertTrue(ex.getMessage().contains("Byte size"));
    }

    @Test
    void copyWithBufferAlsoRespectsGranularity() {
        // Test the overload that takes an explicit buffer
        Memory<long[]> src = longDomain.memoryAllocator().allocateMemory(64);
        Memory<double[]> dst = doubleDomain.memoryAllocator().allocateMemory(64);
        Memory<MemorySegment> buffer = nativeDomain.memoryAllocator().allocateMemory(4096);

        // Should fail: size 12 is not multiple of 8
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MemoryOperations.copy(
                                        longDomain.memoryOperations(),
                                        src,
                                        0,
                                        doubleDomain.memoryOperations(),
                                        dst,
                                        0,
                                        12, // not multiple of 8
                                        buffer));

        assertTrue(ex.getMessage().contains("Byte size"));
    }

    // ========== Helpers ==========

    private MemoryDomain<?> getDomain(String type) {
        return switch (type) {
            case "bytes" -> byteDomain;
            case "shorts" -> shortDomain;
            case "ints" -> intDomain;
            case "longs" -> longDomain;
            case "floats" -> floatDomain;
            case "doubles" -> doubleDomain;
            case "native" -> nativeDomain;
            case "buffer" -> bufferDomain;
            case "direct" -> directBufferDomain;
            default -> throw new IllegalArgumentException("Unknown domain: " + type);
        };
    }

    @SuppressWarnings("unchecked")
    private static <B> void writeByte(MemoryDomain<B> ctx, Memory<?> mem, long offset, byte v) {
        ctx.directAccess().writeByte((Memory<B>) mem, offset, v);
    }

    @SuppressWarnings("unchecked")
    private static <B> byte readByte(MemoryDomain<B> ctx, Memory<?> mem, long offset) {
        return ctx.directAccess().readByte((Memory<B>) mem, offset);
    }

    @SuppressWarnings("unchecked")
    private static <S, D> void copyUntyped(
            MemoryDomain<S> srcCtx,
            Memory<?> src,
            long srcOffset,
            MemoryDomain<D> dstCtx,
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

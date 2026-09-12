package com.qxotic.jota.memory.internal;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.ScopedMemory;
import com.qxotic.jota.memory.ScopedMemoryAllocator;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class UnsafeAllocatorTest {

    static final ScopedMemoryAllocator<MemorySegment> unsafeAllocator =
            MemoryAllocators.newScopedArena();

    @Test
    void testAllocateMemory() {
        try (ScopedMemory<MemorySegment> memory = unsafeAllocator.allocateMemory(100)) {
            assertNotNull(memory);
            assertEquals(100, memory.byteSize());
            assertFalse(memory.isReadOnly());
            assertTrue(memory.device().belongsTo(DeviceType.PANAMA));
            assertNotNull(memory.base());
        }
    }

    @Test
    void testBadAlignment() {
        for (long badAlignment : new long[] {-1, 0, 3, 17, 69}) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> unsafeAllocator.allocateMemory(128, badAlignment));
        }
    }

    @Test
    void negativeSizesAreRejectedBeforeAllocatingAlignmentPadding() {
        try (var arena = MemoryAllocators.newScopedArena()) {
            for (long size : new long[] {-1, -64, Long.MIN_VALUE}) {
                // The largest alignment would cause allocation to fail before reinterpret:
                // asserting the size error also guards the ordering of validation and malloc.
                for (long alignment : new long[] {1, 8, 1 << 20, 1L << 62}) {
                    var failure =
                            assertThrows(
                                    IllegalArgumentException.class,
                                    () -> arena.allocateMemory(size, alignment));
                    assertEquals("negative byte size: " + size, failure.getMessage());
                }
            }
        }
    }

    @Test
    void alignmentPaddingOverflowIsRejected() {
        try (var arena = MemoryAllocators.newScopedArena()) {
            assertThrows(ArithmeticException.class, () -> arena.allocateMemory(Long.MAX_VALUE, 2));
            assertThrows(
                    ArithmeticException.class, () -> arena.allocateMemory(Long.MAX_VALUE - 62, 64));
        }
    }

    @Test
    void zeroAndNonzeroAllocationsKeepTheirSizeAndAlignment() {
        try (var arena = MemoryAllocators.newScopedArena()) {
            for (long size : new long[] {0, 1, 7, 100}) {
                for (long alignment : new long[] {1, 2, 64, 4096}) {
                    try (var memory = arena.allocateMemory(size, alignment)) {
                        assertEquals(size, memory.byteSize());
                        assertEquals(0, memory.base().address() % alignment);
                        if (size > 0) {
                            memory.base().set(ValueLayout.JAVA_BYTE, size - 1, (byte) 42);
                            assertEquals(42, memory.base().get(ValueLayout.JAVA_BYTE, size - 1));
                        }
                    }
                }
            }
        }
    }

    @Test
    void testDevice() {
        assertTrue(unsafeAllocator.device().belongsTo(DeviceType.PANAMA));
    }

    @Test
    void testDoubleCloseFails() {
        ScopedMemory<MemorySegment> memory = unsafeAllocator.allocateMemory(64);
        memory.close();
        assertThrows(IllegalStateException.class, memory::close);
    }
}
